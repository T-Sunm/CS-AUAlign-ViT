"""Target-only evaluation for a completed POST adaptation run."""

import torch


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
    total = int(confusion.sum())
    if not total:
        raise ValueError("Evaluation loader is empty")
    support = confusion.sum(dim=1)
    correct = confusion.diag()
    return {
        "samples": total,
        "accuracy": correct.sum().item() / total,
        "per_class_accuracy": [
            correct[index].item() / support[index].item() if support[index] else None
            for index in range(num_classes)
        ],
        "confusion_matrix": confusion.tolist(),
    }
