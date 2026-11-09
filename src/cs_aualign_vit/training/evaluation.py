"""Target-only evaluation for a completed POST adaptation run."""

import torch


def metrics_from_confusion(confusion):
    """Return finite classification metrics from a square confusion matrix."""
    confusion = torch.as_tensor(confusion, dtype=torch.long)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("Confusion matrix must be square")
    if (confusion < 0).any():
        raise ValueError("Confusion matrix counts must be non-negative")
    total = int(confusion.sum())
    if not total:
        raise ValueError("Confusion matrix is empty")

    correct = confusion.diag().double()
    support = confusion.sum(dim=1).double()
    predicted = confusion.sum(dim=0).double()
    precision = torch.where(predicted > 0, correct / predicted, 0.0)
    recall = torch.where(support > 0, correct / support, 0.0)
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2 * precision * recall / denominator, 0.0)
    return {
        "samples": total,
        "accuracy": float(correct.sum() / total),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / total),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_accuracy": recall.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


@torch.no_grad()
def evaluate_target(model, prototype_banks, loader, device, *, num_classes=7):
    model.eval()
    prototype_banks.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for images, labels in loader:
        labels = labels.to(device)
        features = model.forward_target_features(images.to(device))
        predictions = prototype_banks.target_logits(features).argmax(dim=1)
        confusion += torch.bincount(
            (labels * num_classes + predictions).cpu(), minlength=num_classes**2
        ).reshape(num_classes, num_classes)
    if not int(confusion.sum()):
        raise ValueError("Evaluation loader is empty")
    return metrics_from_confusion(confusion)
