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
        elif loss == "MSE":
            loss_functions.append(nn.MSELoss(reduction="sum"))
        elif loss == "AdaptiveWingLoss":
            loss_functions.append(AdaptiveWingLoss())
        else:
            raise Exception("Not implemented other losses besides BCELoss")
        
    return loss_functions


class BCETverskyLoss(nn.Module):
    def __init__(self, smooth=1.0, alpha=0.5, beta=0.5, coef=0.3):
        super(BCETverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.coef = coef

    def forward(self, input, target):
        """
        Parameters
        - `input` : B, H, W, 1
        - `target` : B, H, W, 1
        """

        input = F.sigmoid(input)
        B, H, W, _ = input.size()

        # B, H, W = input.size()
        input = einops.rearrange(input, 'b h w 1 -> b (h w)')
        target = einops.rearrange(target, 'b h w 1 -> b (h w)')

        t_p = (input * target).sum(dim=-1)
        f_p = ((1-target) * input).sum(dim=-1)
        f_n = (target * (1-input)).sum(dim=-1)
        tversky = (t_p + self.smooth) / (t_p + self.alpha*f_p + self.beta*f_n + self.smooth)
        
        loss = 1 - tversky
        return loss.sum()

class AdaptiveWingLoss(nn.Module):
    """
    Time    : 2019/9/9 
    Author  : Elliott Zheng  
    Email   : admin@hypercube.top
    """
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        B, H, W = pred.size()
        pred = pred.reshape(B, 1, H, W)
        target = target.reshape(B, 1, H, W)

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) * (2*H*W) / (len(loss1) + len(loss2))