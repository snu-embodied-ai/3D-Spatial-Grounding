import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from scheduler.CosineAnnealingWarmupRestarts_jskim import CosineAnnealingWarmUpRestarts as CAWarmup_JS
from scheduler.CosineAnnealingWarmupRestarts_katsura import CosineAnnealingWarmupRestarts as CAWarmup_katsura

def create_optimizer_and_scheduler(param_groups, 
                                   optimizer_name: str, 
                                   optimizer_kwargs: dict, 
                                   scheduler_name: str,
                                   scheduler_kwargs: dict,):
    optimizer = getattr(optim, optimizer_name)(param_groups, **optimizer_kwargs)

    if scheduler_name == "CosineAnnealingWarmUpRestarts_jskim":
        scheduler = CAWarmup_JS(optimizer, 
                                **scheduler_kwargs["CosineAnnealingWarmUpRestarts_jskim"])
    elif scheduler_name == "CosineAnnealingWarmUpRestarts_katsura":
        scheduler = CAWarmup_katsura(optimizer, 
                                     **scheduler_kwargs["CosineAnnealingWarmUpRestarts_katsura"])
    elif scheduler_name == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, 
                                        **scheduler_kwargs["StepLR"])
    elif scheduler_name == "None":
        scheduler = None
    else:
        raise Exception("Other schedulers are not considered yet besides custom CosineAnnealingWarmUpRestarts")
    
    return optimizer, scheduler