import glob
import os
import random
from collections import defaultdict
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np
import pandas as pd
import pickle
import open3d as o3d
from plyfile import PlyData

from data.utils import load_json
from data.constants.scannet import (
    CLASS_LABELS as SCANNET_CLASSES, 
    VALID_CLASS_IDS as SCANNET_CLASS_IDS, 
    IGNORE_LABELS as SCANNET_IGNORE_LABELS)
from data.constants.dataset_consts import TASKS, TASK_TYPE, DATASETS

from accelerate.logging import get_logger

# TODO: STUDY LOGGER
logger = get_logger(__name__)

class SegPointDatasetBase(Dataset):

    def __init__(self,
                 instructions_cfg_path: str):
        super().__init__()
        self.instructions = OmegaConf.load(instructions_cfg_path)

        self.subsample = False
        self.augmentations = []

    def preprocess_pcd(self, pcd: np.ndarray):
        """
        Preprocess the point clouds (Normalization, Subsampling, Augmentations)

        Parameters
        ---
        pcd: np.ndarray
            Point Cloud of the scene including xyz and all features (rgb, ..)
        
        Returns
        ---
        processed_pcd: np.ndarray
            Preprocessed pointcloud (normalized, subsampled, augmented, ...)
        """
        # 1. Normalize the points
        pcd[:, :3] = pcd[:, :3] - pcd[:, :3].mean(0)
        max_dist = np.sqrt((pcd[:,:3]**2).sum(1)).max()

        # Taking care of tiny point clouds
        if max_dist < 1e-6:
            max_dist = 1
        pcd[:, :3] = pcd[:, :3] / max_dist

        # 2. Normalize colors ([0,1] -> [-1,1])
        pcd[:, 3:6] = pcd[:, 3:6] * 2.0 - 1.0

        # 2. Subsample points
        if self.subsample:
            pass

        # 3. Augmentations
        for aug in self.augmentations:
            pass

        return pcd
    
    def per_cat_sample(self, pcd: np.ndarray,
                           labels: np.ndarray,
                           num_samples: int,
                           shuffle: bool = True):
        """
        Sample points per category using FPS (farthest point sampling)

        Parameters
        ---
        pcd: np.ndarray
            Pointcloud data containing xyz and other features (rgb, ...)
        labels: np.ndarray
            Label per point

        Returns
        ---
        all_pcd: torch.Tensor
            Sampled pointcloud data
        all_labels: torch.Tensor
            Labels of sampled points
        """
        rng = np.random.default_rng()

        unique_labels = np.unique(labels)

        all_pcd = []
        all_labels = []

        for category in unique_labels:
            is_cur_cat = labels == category
            cur_cat_pcd = pcd[is_cur_cat]

            if cur_cat_pcd.shape[0] > num_samples:
                # FPS
                cur_cat_pcd = rng.choice(cur_cat_pcd, num_samples, replace=False, axis=0)

            cur_labels = np.full(cur_cat_pcd.shape[0], fill_value=category)

            all_pcd.append(cur_cat_pcd)
            all_labels.append(cur_labels)

        all_pcd = np.concatenate(all_pcd, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if shuffle:
            pcd_w_labels = np.concatenate([all_pcd, all_labels.reshape(-1,1)], axis=1)
            pcd_w_labels = rng.permutation(pcd_w_labels, axis=0)
            all_pcd = pcd_w_labels[:,:-1]
            all_labels = pcd_w_labels[:,-1]

        all_pcd = torch.from_numpy(all_pcd).float()
        all_labels = torch.from_numpy(all_labels)

        return all_pcd, all_labels




"""
Generate Dataloaders TASK-WISE
THEN, a single forward would be applied to same tasks, but training on all tasks within a single epoch. (Refer to Joint Training)
Follow the LEO training pipelines
"""

"""
Write `__getitem__()` based on the label index. Therefore, a single label becomes a single sample, not the scene itself. Same scenes can be included into a single batch, but try to avoid that as you can.
"""

"""
For "Specific Category" semantic segmentation,
    - " .... {category} .... "
Here, a single scene becomes a single sample because a single category is selected randomly for each scene (following the paper)
"""

"""
For "All Category" semantic segmentation,
    - " .... all category .... "
Here, a single scene becomes a single sample because a single scene can generate only one question for this case.
"""

class SemanticSegmentationDataset(SegPointDatasetBase):
    def __init__(self,
                 split: str,
                 segmentation_type: str,
                 dataset_cfg_path: str,
                 instructions_cfg_path: str):
        super().__init__(instructions_cfg_path)

        self.cfg = OmegaConf.load(dataset_cfg_path)["SemanticSegmentation"]
        self.split = split
        self.seg_type = segmentation_type

        # Load the metadata for each dataset (scene id, categories, ...)
        self.all_samples = []
        for dataset_name, dataset_paths in self.cfg.items():
            metadata_dir = Path(dataset_paths.root_dir) / dataset_paths.metadata_dir
            with open(metadata_dir / f"{self.split}.txt", 'r') as f:
                scenes = f.read().split("\n")

            self.all_samples.extend(list(zip([dataset_name] * len(scenes), scenes)))
            
    def __getitem__(self, index):
        # Dictionary with keys - 'scene_id', 'xyz', 'features', 'task', 'task_type', 'mask(labels)', 'category' (for specific), (+ @)
        try:
            dataset_name, scene_id = self.all_samples[index]

            # 1. Get dataset config to get file paths & Get info files
            dataset_paths = self.cfg[dataset_name]
            root_dir = Path(dataset_paths.root_dir)
            scenes_dir = root_dir / dataset_paths.scenes_dir

            # 2. Get pointcloud
            # 2-1. Read PLY
            ply_name = f"{scene_id}{dataset_paths.ply_name}"
            ply_path = scenes_dir / scene_id / ply_name
            if not ply_path.exists():
                raise FileNotFoundError(f"PLY file not found: {ply_path}")
            
            pcd = o3d.io.read_point_cloud(ply_path)
            xyz = np.asarray(pcd.points)
            color = np.asarray(pcd.colors)

            if xyz.shape[0] == 0:
                raise ValueError(f"No points found in {ply_path}")

            if color.shape[0] != xyz.shape[0]:
                raise ValueError(f"Color and XYZ size mismatch in {ply_path}")

            # 2-2. Preprocess features
            features = np.concatenate([xyz, color], axis=-1)
            features = self.preprocess_pcd(features)

            # 2-3. Get labels per point
            label_pcd = PlyData.read(ply_path)
            if "label" not in label_pcd["vertex"].data.dtype.names:
                raise KeyError(f"'label' not found in {ply_path}")
            
            labels = np.asarray(label_pcd['vertex'].data['label'])
            if labels.shape[0] != xyz.shape[0]:
                raise ValueError(f"Labels length does not match points in {ply_path}")

            if dataset_paths.num_samples is not None:
                features, labels = self.per_cat_sample(features, labels, dataset_paths.num_samples, shuffle=True)

            if self.seg_type == "specific":
                if dataset_name == "ScanNet":
                    category = random.choice(SCANNET_CLASS_IDS)
                    mask = (labels == category).float().unsqueeze(dim=0)
                    num_category = 1
                else:
                    # Other dataset
                    raise NotImplementedError(f"Segmentation type '{self.seg_type}' not implemented for {dataset_name}")
            
            elif self.seg_type == "all_categories":
                if dataset_name == "ScanNet":
                    valid = torch.isin(labels, torch.as_tensor(SCANNET_CLASS_IDS, dtype=torch.long))
                    not_onehot_mask = torch.full_like(labels, fill_value=-100)
                    not_onehot_mask[valid] = labels[valid]
                    all_cats = torch.unique(not_onehot_mask)
                    all_cats = all_cats[all_cats != -100]       # Exclude -100

                    if len(all_cats) == 0:
                        raise ValueError(f"No valid categories found in {ply_path}")

                    # Create a binary mask for each category in all_cats
                    # TODO: mask should be in shape of (G, N) (after batching, (B, G, N))
                    # Each (N,) should be a binary mask (after batching (B, N))
                    mask = not_onehot_mask.unsqueeze(0) == all_cats.unsqueeze(1)
                    category = all_cats
                    num_category = len(all_cats)
                else:
                    # Other dataset
                    raise NotImplementedError(f"Segmentation type '{self.seg_type}' not implemented for {dataset_name}")

            else:
                raise ValueError(f"Unknown segmentation type: {self.seg_type}")

            # 3. Generate Data Dictionary
            # Store the indices instead of storing the raw string for converting torch tensors
            data_dict = {
                "dataset_idx": DATASETS.index(dataset_name),
                "xyz": features[:,:3],
                "features": features,
                "task": TASKS.index("SemanticSegmentation"),
                "task_type": TASK_TYPE.index(self.seg_type),
                "mask": mask,
                "category": category,
                "num_category": num_category
            }

            assert isinstance(data_dict, dict), (index, dataset_name, scene_id)
            for k in ["dataset_idx","xyz","features","task","task_type","mask","category","num_category"]:
                assert k in data_dict, (k, index, dataset_name, scene_id)

            return data_dict

        except Exception as e:
            logger.info(f"[ERROR] Dataset index {index} ({dataset_name}/{scene_id}): {e}")
            print(f"[ERROR] Dataset index {index} ({dataset_name}/{scene_id}): {e}")
            return None


    def __len__(self):
        return len(self.all_samples)



class ReferringSegmentationDataset(SegPointDatasetBase):
    def __init__(self):
        super().__init__()

        self.type = None        # specific, all_categories


class Instruct3DSegmentationDataset(SegPointDatasetBase):
    def __init__(self):
        super().__init__()

        self.type = None        # specific, all_categories


def default_collate_fn(batch):
    # rank = dist.get_rank() if dist.is_initialized() else 0
    # print(f"[rank {rank}] batch_len={len(batch)}",
    #       flush=True)
    
    batch = [s for s in batch if s is not None]
    if len(batch) == 0:
        return None
    
    max_num_points = max([sample["xyz"].shape[0] for sample in batch])
    max_num_cats = max([sample["mask"].shape[0] for sample in batch])

    batched_features = torch.full((len(batch), max_num_points, batch[0]["features"].size(1)), fill_value=-100.0)
    batched_masks = torch.full((len(batch), max_num_cats, max_num_points), fill_value=-100.0)
    batched_categories = torch.full((len(batch), max_num_cats), fill_value=-100)
    batched_padding_mask = torch.zeros((len(batch), max_num_points), dtype=bool)
    batched_padding_cats = torch.zeros_like(batched_masks, dtype=bool)
    batched_num_cats = []

    for i, sample in enumerate(batch):
        num_points, _ = sample["features"].size()
        num_cats, _ = sample["mask"].size()

        batched_features[i, :num_points, :] = sample["features"]
        batched_masks[i, :num_cats, :num_points]= sample["mask"]
        batched_categories[i, :num_cats] = sample["category"]
        batched_padding_mask[i, :num_points] = True
        batched_padding_cats[i, :num_cats, :num_points] = True
        batched_num_cats.append(sample["num_category"])

    data_dict = {
        "dataset_idx": batch[0]["dataset_idx"],
        "xyz": batched_features[:,:,:3],
        "features": batched_features,
        "task": batch[0]["task"],
        "task_type": batch[0]["task_type"],
        "mask": batched_masks,
        "category": batched_categories,
        "num_category": torch.tensor(batched_num_cats),
        "padding_mask": batched_padding_mask,
        "valid_gt_mask": batched_padding_cats
    }

    return data_dict

        


def build_dataloaders(task: str,
                      task_type: str,
                      cfg: DictConfig):
    dataloaders = dict()
    
    if task == "SemanticSegmentation":
        train_dataset = SemanticSegmentationDataset(
            split="train",
            segmentation_type=task_type,
            dataset_cfg_path=cfg.dataset_config_path,
            instructions_cfg_path=cfg.instructions_config_path
                    )
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            collate_fn=default_collate_fn,
            sampler=train_sampler
        )

        val_dataset = SemanticSegmentationDataset(
            split="val",
            segmentation_type=task_type,
            dataset_cfg_path=cfg.dataset_config_path,
            instructions_cfg_path=cfg.instructions_config_path
        )
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=cfg.val.batch_size,
            num_workers=cfg.val.num_workers,
            pin_memory=cfg.val.pin_memory,
            collate_fn=default_collate_fn,
            sampler=val_sampler
        )

        test_dataset = SemanticSegmentationDataset(
            split="test",
            segmentation_type=task_type,
            dataset_cfg_path=cfg.dataset_config_path,
            instructions_cfg_path=cfg.instructions_config_path
        )
        test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.test.num_workers,
            pin_memory=cfg.test.pin_memory,
            collate_fn=default_collate_fn,
            sampler=test_sampler
        )
                    
    elif task == "ReferringSegmentation":
        # train_dataset = ReferringSegmentationDataset()
        pass

    elif task == "InstructionSegmentation":
        # train_dataset = Instruct3DSegmentationDataset()
        pass

    else:
        raise Exception("Other tasks are not considered for TRAINING SegPoint!")
    
    return dataloaders