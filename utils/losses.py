import torch
import torch.nn as nn
import torch.nn.functional as F

import einops


"""
Implement arbitrary loss functions in this file
"""

def create_loss_func(loss_types: list,
                     imbalance_weight: int):
    """
    No Reductions applied
    """
    loss_functions = []
    for loss in loss_types:
        if loss == "BCEWithLogitsLoss":
            loss_functions.append(nn.BCEWithLogitsLoss(reduction="sum", pos_weight=torch.tensor(imbalance_weight)))
        elif loss == "KLDivLoss":
            loss_functions.append(nn.KLDivLoss(reduction="sum"))
        elif loss == "BCETverskyLoss":
            loss_functions.append(BCETverskyLoss())
        else:
            raise Exception("Not implemented other losses besides BCELoss")
        
    return loss_functions


class BCETverskyLoss(nn.Module):
    def __init__(self, smooth=1.0, alpha=0.4, beta=0.4, coef=0.5):
        super(BCETverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.coef = coef

    def forward(self, input, target):
        """
        Parameters
        - `input` : B, H, W
        - `target` : B, H, W
        """

        input = F.sigmoid(input)
        _, H, W = input.size()

        # B, H, W = input.size()
        input = einops.rearrange(input, 'b h w -> b (h w)')
        target = einops.rearrange(target, 'b h w -> b (h w)')

        t_p = (input * target).sum(dim=-1)
        f_p = ((1-target) * input).sum(dim=-1)
        f_n = (target * (1-input)).sum(dim=-1)
        tversky = (t_p + self.smooth) / (t_p + self.alpha*f_p + self.beta*f_n + self.smooth)
        
        loss = 1 - tversky
        return loss.sum() * (self.coef*H*W)