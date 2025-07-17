import os, sys
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

import open3d as o3d
import pandas as pd

from .region_division import RegionDivider


class SEGSpatial3DDataset(Dataset):
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

        self.data_dir = data_config["scene_dir"]
        split_dir = data_config["split_dir"]

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

        pcd = o3d.io.read_point_cloud(glob.glob(os.path.join(sample_dir, "tabletop_*.ply"))[0])
        label_pcd = o3d.io.read_point_cloud(glob.glob(os.path.join(sample_dir, "*segmentation.ply"))[0])

        points = torch.tensor(np.asarray(pcd.points))
        features = torch.tensor(np.asarray(points)).clone()
        label = torch.tensor(np.asarray(label_pcd.colors))[:,:1]

        if self.use_rgb:
            rgb = torch.tensor(np.asarray(pcd.colors))
            features = torch.cat((features, rgb), dim=-1)
        
        if self.use_normal:
            pcd.estimate_normals()
            normal = torch.tensor(np.asarray(pcd.normals))
            features = torch.cat((features, normal), dim=-1)


        # ===== 2. Divide the scene into regions =============================================
        divided_xyz, divided_feats, divided_label, mask, centers, scales, rot_mat = self.divider.divide_regions(points, features, label)


        # ==== 3. Pad tokens / Random sample tokens to match the max number of tokens=========
        num_regions, _, _ = divided_xyz.size()
        if num_regions < self.max_num_tokens:
            pad_seq = (0,0, 0,0, 0, self.max_num_tokens - num_regions)
            divided_xyz = F.pad(divided_xyz, pad_seq, 'constant', 0)
            divided_feats = F.pad(divided_feats, pad_seq, 'constant', 0)
            divided_label = F.pad(divided_label, pad_seq, 'constant', 0)
            mask = F.pad(mask, pad_seq[2:], 'constant', 0)
            centers = F.pad(centers, pad_seq[2:], 'constant', 0)
            scales = F.pad(scales, pad_seq[4:], 'constant', 0)
        elif num_regions > self.max_num_tokens:
            selected = np.random.choice(num_regions, self.max_num_tokens, replace=False)
            divided_xyz = divided_xyz[selected]
            divided_feats = divided_feats[selected]
            divided_label = divided_label[selected]
            mask = mask[selected]
            centers = centers[selected]
            scales = scales[selected]
        

        # ==== 4. Output dictionary ==========================================================
        vision_dict = {
            "divided_xyz": divided_xyz.float(),                                     # (num_groups, points_per_region, 3)
            "divided_features": divided_feats.float(),                              # (num_groups, points_per_region, 6+@)
            "divided_labels": divided_label.float(),                                # (num_groups, points_per_region, 1)
            "mask": mask.float(),                                                   # (num_groups, points_per_region)
            "centers": centers.float(),                                             # (num_groups, 3)
            "scales": scales.float(),                                               # (num_groups,)
            "rotation_matrix": torch.from_numpy(rot_mat).float(),                   # (4, 4)
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