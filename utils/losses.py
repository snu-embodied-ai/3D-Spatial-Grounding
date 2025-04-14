import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys

"""
Implement arbitrary loss functions in this file
"""

def create_loss_func(loss_types: list):
    """
    No Reductions applied
    """
    loss_functions = []
    for loss in loss_types:
        if loss == "BCEWithLogitsLoss":
            loss_functions.append(nn.BCEWithLogitsLoss(reduction="none"))
        elif loss == "KLDivLoss":
            raise Exception("Not implemented KLDivLoss yet")
        else:
            raise Exception("Not implemented other losses besides BCEWithLogitsLoss")
        
    return loss_functions