import torch

def compute_metrics(logits, targets, threshold=0.5, smooth=1e-6):
    """
    Computes metrics per-image then averages across the batch.
    Targets should be 0 or 1.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        targets = targets.float()
        
        batch_size = preds.size(0)
        
        # Flatten spatial dims: [N, 1, H, W] -> [N, H*W]
        preds = preds.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        # True Positives, False Positives, False Negatives, True Negatives per image
        tp = (preds * targets).sum(dim=1)
        fp = (preds * (1 - targets)).sum(dim=1)
        fn = ((1 - preds) * targets).sum(dim=1)
        
        # Compute metrics per image
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        
        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + smooth)
        
        dice = (2.0 * intersection) / (2.0 * intersection + fp + fn + smooth)
        
        # Average across the batch
        return {
            "dice": dice.mean().item(),
            "iou": iou.mean().item(),
            "precision": precision.mean().item(),
            "recall": recall.mean().item()
        }
