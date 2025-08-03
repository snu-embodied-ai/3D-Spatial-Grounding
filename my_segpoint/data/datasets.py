import glob
import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle
import open3d as o3d

from data.utils import load_json

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

        # 2. Subsample points
        if self.subsample:
            pass

        # 3. Augmentations
        for aug in self.augmentations:
            pass

        # 4. Convert numpy array to torch
        pcd = torch.from_numpy(pcd).float()

        return pcd
    
    def get_metadata(self,
                     dataset_name: str,
                     task: str,
                     task_type: str,):
        """
        Load the metadata for queried dataset

        1. "Specific Category" Semantic Segmentation
            each sample is a dictionary with keys - 'dataset_name', 'scene_id'
        2. "All Category" Semantic Segmentation
            each sample is a dictionary with keys - 'dataset_name', 'scene_id'

        """
        pass

    def load_scannet200(self):
        pass

    def load_s3dis(self):
        pass

    def load_scanrefer(self):
        pass

    def load_sr3d(self):
        pass

    def load_nr3d(self):
        pass

    def load_multi3drefer(self):
        pass



"""
Generate Dataloaders TASK-WISE
THEN, a single forward would be applied to same tasks, but training on all tasks within a single epoch. (Refer to Joint Training)
Follow the LEO training pipelines
"""

"""
Write `__getitem__()` based on the label index. Therefore, a single label becomes a single sample, not the scene itself. Same scenes can be included into a single batch, but try to avoid that as you can.
"""
# Assuming "Specific Category" semantic segmentation,
#     - " .... {category} .... "
# Here, a single category becomes a single sample, therefore a single scene can generate multiple samples.
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

        self.cfg = OmegaConf.load(dataset_cfg_path)
        self.split = split
        self.seg_type = segmentation_type

        self.type = None        # specific, all_categories

        # Load the metadata for each dataset (scene id, categories, ...)
        self.all_samples = []
        self.category_mapping_files = dict()
        for dataset_name, dataset_paths in self.cfg.items():
            metadata = self.get_metadata(dataset_name=dataset_name,
                                         task="SemanticSegmentation",
                                         task_type=self.seg_type)
            self.all_samples.append(metadata)

            category_mapping_file = Path(dataset_paths.root_dir) / dataset_paths.category_mapping_file_name
            category_mapping = pd.read_csv(category_mapping_file,
                                           sep="\t" if category_mapping_file.endswith('.tsv') else ",")
            self.category_mapping_files[dataset_name] = category_mapping

            
    def __getitem__(self, index):
        # return super().__getitem__(index)
        # Dictionary with keys - 'scene_id', 'xyz', 'features', 'task', 'task_type', 'mask(labels)', 'category' (for specific), (+ @)
        data_dict = dict()

        dataset_name, scene_id = self.all_samples[index]

        # 1. Get dataset config to get file paths & Get info files
        dataset_paths = self.cfg[dataset_name]
        root_dir = Path(dataset_paths.root_dir)

        seg_path = root_dir / scene_id / dataset_paths.segments_file_name
        semseg_path = root_dir / scene_id / dataset_paths.semantics_file_name
        cat_mapping = self.category_mapping_files[dataset_name]

        seg_data = load_json(seg_path)
        semseg_data = load_json(semseg_path)

        # Select category (not object, since multiple objects can exist in a scene)
        objects_in_scene = semseg_data["segGroups"]

        # 2. Get pointcloud
        ply_path = root_dir / scene_id / dataset_paths.ply_name
        pcd = o3d.io.read_pointcloud(ply_path)
        xyz = np.asarray(pcd.points)
        color = np.asarray(pcd.colors)
        features = np.concatenate([xyz, color], axis=-1)

        

        



        pass

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