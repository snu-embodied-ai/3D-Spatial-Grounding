import os, sys
import glob

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import gaussian_blur
import copy

import numpy as np

import open3d as o3d
import pandas as pd

from .region_division import RegionDivider

def spatial_collate_fn(batch):
    max_width = max([sample[0]["heatmap_label"].shape[0] for sample in batch])
    max_height = max([sample[0]["heatmap_label"].shape[1] for sample in batch])

    batched_divided_coords, batched_divided_features, batched_mask, batched_centers, batched_rotation_matrix = [], [], [], [], []

    batched_idx, batched_input_ids, batched_padding_mask, batched_position_ids = [], [], [], []

    batched_heatmap_label = torch.zeros((len(batch), max_width, max_height))
    batched_heatmap_masks = torch.zeros_like(batched_heatmap_label)
    batched_heatmap_centers = torch.zeros((len(batch), max_width, max_height, 2))

    batched_occupancy_label = torch.zeros((len(batch), max_width, max_height))

    for i, (v_sample, l_sample) in enumerate(batch):
        width, height = v_sample["heatmap_label"].shape

        batched_divided_coords.append(v_sample["divided_xyz"])
        batched_divided_features.append(v_sample["divided_features"])
        batched_mask.append(v_sample["mask"])
        batched_centers.append(v_sample["centers"])
        batched_rotation_matrix.append(v_sample["rotation_matrix"])

        batched_idx.append(l_sample["index"])
        batched_input_ids.append(l_sample["input_ids"])
        batched_padding_mask.append(l_sample["padding_mask"])
        batched_position_ids.append(l_sample["position_ids"])

        batched_heatmap_label[i, :width, :height] = v_sample["heatmap_label"]
        batched_heatmap_masks[i, :width, :height] = 1
        batched_heatmap_centers[i, :width, :height, :] = v_sample["heatmap_grid_centers"]

        batched_occupancy_label[i, :width, :height] = v_sample["occupancy_label"]


    batched_divided_coords = torch.stack(batched_divided_coords, dim=0)
    batched_divided_features = torch.stack(batched_divided_features, dim=0)
    batched_mask = torch.stack(batched_mask, dim=0)
    batched_centers = torch.stack(batched_centers, dim=0)
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
        "heatmap_grid_centers": batched_heatmap_centers,    # (batch_size, row, col, 2)
        "occupancy_label": batched_occupancy_label,         # (batch_size, row, col)
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
        self.max_num_tokens = data_config["max_num_tokens"]

        self.data_dir = data_config[data_type]["data_dir"]
        # self.use_whole = data_config["use_whole"]

        self.prefix = data_config["prefix"]
        self.descriptions = pd.read_csv(os.path.join(self.data_dir, f"{data_type}_labels.csv"))

        # TODO : MUST add more properties

    def __getitem__(self, index):
        # TODO : COLLATE FUNCTION for DataLoader!!!

        # ===== 1. Load Description ==========================================================
        sample_id = self.descriptions.iloc[index].id
        relation = self.descriptions.iloc[index].description

        description = self.prefix + relation

        description, padding_mask, position_ids = self.tokenizer(description)

        # ==== 2. Load PLY file and heatmap label ============================================
        sample_dir = os.path.join(self.data_dir, sample_id)

        pcd = o3d.io.read_point_cloud(glob.glob(os.path.join(sample_dir, "*.ply"))[0])
        heatmap_label = np.load(glob.glob(os.path.join(sample_dir, "*label*"))[0])
        occupancy_label = np.load(glob.glob(os.path.join(sample_dir, "*occupancy*"))[0])
        width, height = heatmap_label.shape

        occupancy_label = torch.from_numpy(occupancy_label)
        heatmap_label = torch.from_numpy(heatmap_label)

        heatmap_centers = np.indices((width, height)).transpose((1,2,0)) * self.grid_size + (self.grid_size/2)

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


        # ==== 3. Pad tokens / Random sample tokens to match the max number of tokens=========
        num_regions, num_points, _ = divided_xyz.shape
        if num_regions < self.max_num_tokens:
            divided_xyz = np.pad(divided_xyz, ((0, self.max_num_tokens - num_regions), (0,0), (0,0)), 'constant', constant_values=0)
            divided_feats = np.pad(divided_feats, ((0, self.max_num_tokens - num_regions), (0,0), (0,0)), 'constant', constant_values=0)
            mask = np.pad(mask, ((0, self.max_num_tokens - num_regions), (0,0)), 'constant', constant_values=0)
            centers = np.pad(centers, ((0, self.max_num_tokens - num_regions), (0,0)), 'constant', constant_values=0)
        elif num_regions > self.max_num_tokens:
            selected = np.random.choice(num_regions, self.max_num_tokens, replace=False)
            divided_xyz = divided_xyz[selected]
            divided_feats = divided_feats[selected]
            mask = mask[selected]
            centers = centers[selected]
        

        # ==== 4. Output dictionary ==========================================================
        vision_dict = {
            "divided_xyz": torch.from_numpy(divided_xyz).float(),              # (num_groups, points_per_region, 3)
            "divided_features": torch.from_numpy(divided_feats).float(),       # (num_groups, points_per_region, 6+@)
            "mask": torch.from_numpy(mask).float(),                            # (num_groups, points_per_region)
            "centers": torch.from_numpy(centers).float(),                              # (num_groups, 3)
            "rotation_matrix": torch.from_numpy(rot_mat).float(),              # (4, 4)
            "heatmap_label": heatmap_label.reshape(width, height),             # (width, height)
            "heatmap_grid_centers": torch.from_numpy(heatmap_centers).float(), # (width, height, 2)
            "occupancy_label": occupancy_label.reshape(width, height),         # (width, height, 2)
        }

        lang_dict = {
            "index": torch.tensor([int(sample_id)], dtype=torch.int),
            "input_ids": description,                   # (token_size)
            "padding_mask": padding_mask,               # (token_size)
            "position_ids": position_ids,               # (token_size)
        }

        # if self.use_whole:
        #     data_dict["whole_features"] = features      # (num_points, 6+@)

        return vision_dict, lang_dict


    def __len__(self):
        return len(self.descriptions)