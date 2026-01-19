import torch


@torch.no_grad()
def multilabel_accuracy(logits, targets, threshold=0.5):
    """
    Returns:
        overall_acc: scalar
        per_label_acc: (num_labels,)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    correct = (preds == targets).float()

    overall_acc = correct.mean().item()
    per_label_acc = correct.mean(dim=0).cpu().numpy()

    return overall_acc, per_label_acc
