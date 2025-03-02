import torch


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    Compute precision, recall, and F1 score for multi-label classification.

    Args:
        outputs (torch.Tensor): Model predictions (logits).
        targets (torch.Tensor): Ground truth labels.
        threshold (float): Threshold for binary classification.

    Returns:
        dict: Dictionary containing precision, recall, and F1 score.
    """
    # Apply sigmoid and threshold
    preds = (torch.sigmoid(outputs) > threshold).float()

    # True positives, false positives, false negatives
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    # Precision, recall, F1
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item()
    }
