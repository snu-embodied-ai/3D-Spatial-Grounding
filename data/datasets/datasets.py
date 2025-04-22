import os, sys
import glob

import torch
import numpy as np
from torch.utils.data import Dataset
import copy

import yaml

import json
from tqdm import tqdm
import pickle

import open3d as o3d
import pandas as pd

from .region_division import RegionDivider

def spatial_collate_fn(batch):
    max_len = max([sample[0]["mask"].shape[0] for sample in batch])
    _, points_per_group, num_coords = batch[0][0]["divided_xyz"].shape
    num_features = batch[0][0]["divided_features"].shape[-1]
    token_size = batch[0][1]["input_ids"].shape[-1]

    max_width = max([sample[0]["heatmap_label"].shape[0] for sample in batch])
    max_height = max([sample[0]["heatmap_label"].shape[1] for sample in batch])
    
    batched_divided_coords = torch.zeros((len(batch), max_len, points_per_group, num_coords))
    batched_divided_features = torch.zeros((len(batch), max_len, points_per_group, num_features))
    batched_mask = torch.zeros((len(batch), max_len, points_per_group))
    batched_centers = torch.zeros((len(batch), max_len, 3))

    batched_rotation_matrix = []
    batched_idx, batched_input_ids, batched_padding_mask, batched_position_ids = [], [], [], []

    batched_heatmap_label = torch.zeros((len(batch), max_width, max_height))
    batched_heatmap_masks = torch.zeros_like(batched_heatmap_label)
    batched_heatmap_centers = torch.zeros((len(batch), max_width, max_height, 2))

    for i, (v_sample, l_sample) in enumerate(batch):
        num_groups, _, _ = v_sample["divided_xyz"].shape
        width, height = v_sample["heatmap_label"].shape

        batched_divided_coords[i, :num_groups] = v_sample["divided_xyz"]
        batched_divided_features[i, :num_groups] = v_sample["divided_features"]
        batched_mask[i, :num_groups] = v_sample["mask"]
        batched_centers[i, :num_groups] = v_sample["centers"]

        batched_rotation_matrix.append(v_sample["rotation_matrix"])

        batched_idx.append(l_sample["index"])
        batched_input_ids.append(l_sample["input_ids"])
        batched_padding_mask.append(l_sample["padding_mask"])
        batched_position_ids.append(l_sample["position_ids"])

        batched_heatmap_label[i, :width, :height] = v_sample["heatmap_label"]
        batched_heatmap_masks[i, :width, :height] = 1
        batched_heatmap_centers[i, :width, :height, :] = v_sample["heatmap_grid_centers"]


    batched_rotation_matrix = torch.stack(batched_rotation_matrix, dim=0)

    batched_idx = torch.stack(batched_idx, dim=0)
    batched_input_ids = torch.stack(batched_input_ids, dim=0)
    batched_padding_mask = torch.stack(batched_padding_mask, dim=0)
    batched_position_ids = torch.stack(batched_position_ids, dim=0)


    vision_dict = {
        "divided_xyz": batched_divided_coords,              # (batch_size, num_groups, num_points, 3)
        "divided_features": batched_divided_features,       # (batch_size, num_groups, num_points, 6+@)
        "mask": batched_mask,                               # (batch_size, num_groups, num_points)
        "centers": batched_centers,                         # (batch_size, num_groups, 3)
        "rotation_matrix": batched_rotation_matrix,         # (batch_size, 4, 4)
        "heatmap_label": batched_heatmap_label,             # (batch_size, row, col)
        "heatmap_masks": batched_heatmap_masks,             # (batch_size, row, col)
        "heatmap_grid_centers": batched_heatmap_centers     # (batch_size, row, col, 2)
    }
    
    lang_dict = {
        "index": batched_idx,                               # (batch_size, 1)
        "input_ids": batched_input_ids,                     # (batch_size, token_size)
        "padding_mask": batched_padding_mask,               # (batch_size, token_size)
        "position_ids": batched_position_ids,               # (batch_size, token_size)
    }

    return vision_dict, lang_dict


class Spatial3DDataset(Dataset):
    def __init__(self, data_config, tokenizer, data_type):
        """

        Written based on my toy dataset format
        
        - `data_config` : configuration file for datasets
        - `type` : Type of the dataset. "train", "val", "test" are valid arguments

        """
        self.config = data_config
        self.tokenizer = tokenizer
        self.dataset_type = data_type

        self.divider = RegionDivider(**self.config["Region"])

        self.grid_size = data_config["grid_size"]
        self.use_rgb = data_config["use_rgb"]
        self.use_normal = data_config["use_normal"]

        self.data_dir = data_config[data_type]["data_dir"]
        # self.use_whole = data_config["use_whole"]
        
        self.descriptions = pd.read_csv(os.path.join(self.data_dir, f"{data_type}_labels.csv"))

        # TODO : MUST add more properties

    def __getitem__(self, index):
        # TODO : COLLATE FUNCTION for DataLoader!!!

        # ===== 1. Load Description ==========================================================
        sample_id = self.descriptions.iloc[index].id
        description = self.descriptions.iloc[index].description

        description, padding_mask, position_ids = self.tokenizer(description)

        # ==== 2. Load PLY file and heatmap label ============================================
        sample_dir = os.path.join(self.data_dir, sample_id)

        pcd = o3d.io.read_point_cloud(glob.glob(os.path.join(sample_dir, "*.ply"))[0])
        heatmap_label = np.load(glob.glob(os.path.join(sample_dir, "*label*"))[0])

        heatmap_centers = np.indices(heatmap_label.shape).transpose((1,2,0)) * self.grid_size + (self.grid_size/2)

        points = np.asarray(pcd.points)
        features = copy.deepcopy(points)

        if self.use_rgb:
            rgb = np.asarray(pcd.colors)
            features = np.concatenate((features, rgb), axis=-1)
        
        if self.use_normal:
            pcd.estimate_normals()
            normal = np.asarray(pcd.normals)
            features = np.concatenate((features, normal), axis=-1)


        # ===== 2. Divide the scene into regions =============================================
        divided_xyz, divided_feats, mask, centers, rot_mat = self.divider.divide_regions(points, features)
        

        # ==== 3. Output dictionary ==========================================================
        vision_dict = {
            "divided_xyz": torch.from_numpy(divided_xyz).float(),              # (num_groups, points_per_region, 3)
            "divided_features": torch.from_numpy(divided_feats).float(),       # (num_groups, points_per_region, 6+@)
            "mask": torch.from_numpy(mask).float(),                            # (num_groups, points_per_region)
            "centers": torch.from_numpy(centers),                              # (num_groups, 3)
            "rotation_matrix": torch.from_numpy(rot_mat).float(),              # (4, 4)
            "heatmap_label": torch.from_numpy(heatmap_label).int(),            # (width, height)
            "heatmap_grid_centers": torch.from_numpy(heatmap_centers).int(),            # (width, height, 2)
        }

        lang_dict = {
            "index": torch.tensor([int(sample_id)], dtype=torch.int),
            "input_ids": description,                   # (token_size)
            "padding_mask": padding_mask,           # (token_size)
            "position_ids": position_ids,               # (token_size)
        }

        # if self.use_whole:
        #     data_dict["whole_features"] = features      # (num_points, 6+@)

        return vision_dict, lang_dict


    def __len__(self):
        return len(self.descriptions)