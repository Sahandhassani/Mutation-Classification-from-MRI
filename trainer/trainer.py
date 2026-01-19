import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from .metrics import multilabel_accuracy
from .utils import save_json, save_csv, get_timestamp
from collections import defaultdict

class Trainer:
    def __init__(
        self,
        model,
        model_name,
        device,
        criterion,
        optimizer,
        experiment_dir,
        label_names,        
        scheduler=None
    ):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment_dir = experiment_dir

        self.label_names = label_names   
        self.history = []
        
    def multilabel_accuracy(logits, targets, threshold=0.5):
        """
        logits: Tensor [N, C]
        targets: Tensor [N, C]
        """
        preds = (torch.sigmoid(logits) > threshold).float()

        correct = (preds == targets).float()
        per_label_acc = correct.mean(dim=0)        # [C]
        mean_acc = per_label_acc.mean().item()     # scalar

        return mean_acc, per_label_acc
        
    def aggregate_patient_predictions(logits, targets, patient_ids):
        """
        logits: Tensor [num_slices, num_labels]
        targets: Tensor [num_slices, num_labels]
        patient_ids: list[str]
        """

        patient_logits = defaultdict(list)
        patient_targets = {}

        for logit, target, pid in zip(logits, targets, patient_ids):
            patient_logits[pid].append(logit)
            patient_targets[pid] = target  # same for all slices

        agg_logits = []
        agg_targets = []

        for pid in patient_logits:
            agg_logits.append(torch.stack(patient_logits[pid]).mean(dim=0))
            agg_targets.append(patient_targets[pid])

        return torch.stack(agg_logits), torch.stack(agg_targets)

    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        all_logits = []
        all_targets = []

        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for imgs, targets in tqdm(
                loader,
                desc="Training" if train else "Validating",
                leave=False
            ):
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # 🔥 DO NOT resize here (already done in dataset)

                logits = self.model(imgs)
                loss = self.criterion(logits, targets)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                batch_size = imgs.size(0)
                total_loss += loss.item() * batch_size

                all_logits.append(logits.detach().cpu())
                all_targets.append(targets.detach().cpu())

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)

        avg_loss = total_loss / len(loader.dataset)
        overall_acc, per_label_acc = multilabel_accuracy(logits, targets)

        return avg_loss, overall_acc, per_label_acc

    def train(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        num_epochs=20
    ):
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")

            # ---- TRAIN ----
            train_loss, train_acc, train_label_acc = self._run_epoch(
                train_loader, train=True
            )

            # ---- VALIDATION ----
            val_loss, val_acc, val_label_acc = self._run_epoch(
                val_loader, train=False
            )

            if self.scheduler:
                self.scheduler.step(val_loss)

            # ---- RECORD METRICS ----
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy_mean": train_acc,
                "val_loss": val_loss,
                "val_accuracy_mean": val_acc,
                "timestamp": get_timestamp(),
            }

            # ---- PER-LABEL TRAIN ACC ----
            for i, name in enumerate(self.label_names):
                epoch_record[f"train_accuracy_{name}"] = float(train_label_acc[i])
                epoch_record[f"val_accuracy_{name}"]   = float(val_label_acc[i])

            self.history.append(epoch_record)

            # ---- PRINT ----
            print(f"Train Loss: {train_loss:.4f} | Mean Acc: {train_acc:.4f}")
            for i, name in enumerate(self.label_names):
                print(f"  └─ Train {name}: {train_label_acc[i]:.4f}")

            print(f"Val   Loss: {val_loss:.4f} | Mean Acc: {val_acc:.4f}")
            for i, name in enumerate(self.label_names):
                print(f"  └─ Val {name}: {val_label_acc[i]:.4f}")

            # ---- SAVE BEST MODEL ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    f"{self.experiment_dir}/trained_model.pt"
                )
                print("✓ Best model saved")

        # ---- TEST ----
        test_results = None
        if test_loader:
            test_loss, test_acc, test_label_acc = self._run_epoch(
                test_loader, train=False
            )

            test_results = {
                "test_loss": test_loss,
                "test_accuracy_mean": test_acc,
                "test_accuracy_per_label": {
                    self.label_names[i]: float(test_label_acc[i])
                    for i in range(len(self.label_names))
                }
            }

        # ---- SAVE RESULTS ----
        results = {
            "model_name": self.model_name,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "criterion": self.criterion.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "epochs": num_epochs,
            "best_val_loss": best_val_loss,
            "final_val_accuracy_mean": val_acc,
            "final_val_accuracy_per_label": {
                self.label_names[i]: float(val_label_acc[i])
                for i in range(len(self.label_names))
            },
            "test_results": test_results,
        }

        save_json(results, f"{self.experiment_dir}/results.json")
        save_csv(self.history, f"{self.experiment_dir}/history.csv")

        print("\n✓ Training complete")
        print(f"✓ Results saved to {self.experiment_dir}")
