import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten predictions and targets
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice_score

class CombinedLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_dice=1.0, smooth=1.0):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        # Targets must be float for BCEWithLogitsLoss
        targets = targets.float()
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.w_bce * loss_bce + self.w_dice * loss_dice

def get_loss_function(name="combined", **kwargs):
    if name == "combined":
        return CombinedLoss(**kwargs)
    elif name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "dice":
        return SoftDiceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function {name}")
