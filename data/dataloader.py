import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import os, sys
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
sys.path.extend([ROOT_DIR, CUR_DIR])

from data.datasets.datasets import spatial_collate_fn, Spatial3DDataset

def create_dataloader(data_cfg: dict, 
                      tokenizer: object):
    """
    Create Dataloaders.

    Input
    ------------------------------------------------
    - `data_cfg`: Configurations for the dataset.
    - `tokenizer`: tokenizer for tokenizing the language descriptions

    Return
    ------------------------------------------------
    - (Training) tuple(`train_data_loader`, `val_data_loader`, `test_data_loader`)
    - (Test Only) tuple(None, None, `test_data_loader`)
    """
    
    if data_cfg["only_test"]:
        train_data_loader, val_data_loader = None, None
    else:
        data_cfg["train"]["data_dir"] = os.path.join(ROOT_DIR, data_cfg["train"]["data_dir"])
        data_cfg["val"]["data_dir"] = os.path.join(ROOT_DIR, data_cfg["val"]["data_dir"])

        train_dataset = Spatial3DDataset(data_cfg, tokenizer, data_type="train")
        val_dataset = Spatial3DDataset(data_cfg, tokenizer, data_type="val")

        # train_sampler = DistributedSampler(train_dataset, shuffle=True)
        # val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=data_cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=data_cfg["train"]["num_workers"],
            collate_fn=spatial_collate_fn,
            # sampler=train_sampler,
            pin_memory=data_cfg["train"]["pin_memory"]
        )
        val_data_loader = DataLoader(
            dataset=val_dataset,
            batch_size=data_cfg["val"]["batch_size"],
            shuffle=False,
            num_workers=data_cfg["val"]["num_workers"],
            collate_fn=spatial_collate_fn,
            # sampler=val_sampler,
            pin_memory=data_cfg["val"]["pin_memory"]
        )
        
    data_cfg["test"]["data_dir"] = os.path.join(ROOT_DIR, data_cfg["test"]["data_dir"])

    test_dataset = Spatial3DDataset(data_cfg, tokenizer, data_type="test")
    # test_sampler = DistributedSampler(test_dataset, shuffle=False)

    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=data_cfg["test"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["test"]["num_workers"],
        collate_fn=spatial_collate_fn,
        # sampler=test_sampler,
        pin_memory=data_cfg["test"]["pin_memory"]
    )

    return train_data_loader, val_data_loader, test_data_loader