from preprocessing.kmeans_remover import process_series_with_kmeans_and_save
from utils.mri_utils import load_dicom_series_safe
from preprocessing.build_mri_mutation_df import build_merged_mri_mutation_df
import os
from preprocessing.build_datasets import build_train_val_test_datasets
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from trainer.trainer import Trainer
from trainer.utils import create_experiment_dir

from models.resnet.resnet158 import ResNet158
from models.resnet.SENet import SENet

from models.lstm.VisionLstm import VisionLSTM

from models.transformers.ViT import VisionTransformer
from models.transformers.pixel_transformer import PixelTransformer2D
from models.transformers.swin2d import SwinTransformer2D

from models.cnn.convnext_v2_2d import ConvNeXtV2_2D
from models.cnn.hybrid_cnn_global import HybridCNNGlobal
from models.cnn.ShiftwiseConv import ShiftwiseCNN
from models.cnn.alexnet_2d import AlexNet2D



# Build dataframe (metadata only)
df = build_merged_mri_mutation_df(
    mri_root="data/MRI/raw",
    mutation_csv="Data/mutation/TCGA_GBM_mutation_labels_02.csv"
)

raw_root = "data/MRI/raw"
output_root = "data/processed/black_ratio"

for _, row in df.iterrows():
    patient = row["Patient"]
    session = row["Session"]

    session_dir = os.path.join(raw_root, patient, session)
    if not os.path.isdir(session_dir):
        continue

    for series_name in os.listdir(session_dir):
        series_dir = os.path.join(session_dir, series_name)
        if not os.path.isdir(series_dir):
            continue

        if not any(f.lower().endswith(".dcm") for f in os.listdir(series_dir)):
            continue

        print(f"▶ Processing {patient} | {series_name}")

        process_series_with_kmeans_and_save(
            series_dir=series_dir,
            patient=patient,
            session=session,
            output_root=output_root,
            percentile=5,
            n_clusters=3,
        )


merged_df = build_merged_mri_mutation_df(
    mri_root="data/MRI/raw",
    mutation_csv="Data/mutation/TCGA_GBM_mutation_labels_02.csv"
)

label_cols = ["IDH_mut", "TP53_mut", "EGFR_mut", "PTEN_mut"]

train_ds, val_ds, test_ds = build_train_val_test_datasets(
    merged_df=merged_df,
    processed_root="data/processed/black_ratio",
    label_cols=label_cols, 
    max_slices_per_patient=1000,
)


batch_size = 32
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

img, label = train_ds[0]

print(img.shape)     # (1, H, W)
print(label.shape)   # (num_labels,)

def get_models(num_labels=4):
    return [
        ("AlexNet2D", AlexNet2D(num_labels=num_labels)),
        ("SENet", SENet(num_labels=num_labels)),
        ("ConvNeXtV2_2D", ConvNeXtV2_2D(in_chans=1, num_classes=num_labels)),
        ("VisionLSTM", VisionLSTM(img_size=224, patch_size=16, in_chans=1, embed_dim=256, hidden_dim=256, depth=6, num_labels=num_labels)),
        ("ViT",  VisionTransformer(num_labels=num_labels)),
        ("ResNet158", ResNet158(num_labels=num_labels)),
        #("ShiftwiseCNN", ShiftwiseCNN(in_channels=1, num_classes=num_labels)),
        #("PixelTransformer2D", PixelTransformer2D(img_size=224, in_channels=1, embed_dim=256, depth=8, num_heads=8, num_classes=num_labels)),
        #("SwinTransformer2D", SwinTransformer2D(img_size=224, in_chans=1, num_classes=num_labels)),
        #("HybridCNNGlobal", HybridCNNGlobal(in_chans=1, num_classes=num_labels)),
    ]

def train_all_models(
    train_loader,
    val_loader,
    test_loader,
    num_epochs=25,
    lr=1e-4,
    weight_decay=1e-5,
    base_exp_dir="experiments"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()

    models = get_models(num_labels=4)

    for model_name, model in models:
        print("\n" + "=" * 80)
        print(f"🚀 Training model: {model_name}")
        print("=" * 80)

        model = model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        exp_dir = create_experiment_dir(base_exp_dir, model_name)

        trainer = Trainer(
            model=model,
            model_name=model_name,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            experiment_dir=exp_dir,
            label_names=["IDH", "TP53", "EGFR", "PTEN"]
        )

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=num_epochs
        )

        # ---- Free memory between models ----
        del model
        torch.cuda.empty_cache()

train_all_models(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_epochs=10
)        
