import glob
import os
import random
from collections import defaultdict
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import pandas as pd
import pickle
import open3d as o3d
from plyfile import PlyData

from data.utils import load_json
from data.constants.scannet import CLASS_LABELS, VALID_CLASS_IDS, IGNORE_LABELS as SCANNET_CLASSES, SCANNET_CLASS_IDS, SCANNET_IGNORE_LABELS
from my_segpoint.data.constants.dataset_consts import TASKS, TASK_TYPE, SCENE_IDS, DATASETS

"""
TODO: COLLATE FUNCTION!!!
"""

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

        # 4. Convert numpy array to torch
        pcd = torch.from_numpy(pcd).float()

        return pcd
    
    # def get_metadata(self,
    #                  dataset_name: str,
    #                  task: str,
    #                  task_type: str,):
    #     """
    #     Load the metadata for queried dataset

    #     1. "Specific Category" Semantic Segmentation
    #         each sample is a dictionary with keys - 'dataset_name', 'scene_id'
    #     2. "All Category" Semantic Segmentation
    #         each sample is a dictionary with keys - 'dataset_name', 'scene_id'

    #     """
    #     pass

    # def load_scannet200(self):
    #     pass

    # def load_s3dis(self):
    #     pass

    # def load_scanrefer(self):
    #     pass

    # def load_sr3d(self):
    #     pass

    # def load_nr3d(self):
    #     pass

    # def load_multi3drefer(self):
    #     pass



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

        # Load the metadata for each dataset (scene id, categories, ...)
        self.all_samples = []
        self.category_mapping_files = dict()
        for dataset_name, dataset_paths in self.cfg.items():
            metadata_dir = Path(dataset_paths) / dataset_paths.metatdata_dir
            with open(metadata_dir / f"{self.split}.txt", 'r') as f:
                scenes = f.read().split("\n")

            # metadata = self.get_metadata(dataset_name=dataset_name,
            #                              task="SemanticSegmentation",
            #                              task_type=self.seg_type)
            # self.all_samples.append(metadata)
            self.all_samples.extend(list(zip([dataset_name] * len(scenes), scenes)))
            

            # category_mapping_file = Path(dataset_paths.root_dir) / dataset_paths.category_mapping_file_name
            # category_mapping = pd.read_csv(category_mapping_file,
            #                                sep="\t" if category_mapping_file.endswith('.tsv') else ",")
            # self.category_mapping_files[dataset_name] = category_mapping

    # def cat_2_int(self,
    #               dataset_name: str,
    #               obj_label: str):
    #     cat_mapping = self.category_mapping_files[dataset_name]
    #     cat_row = cat_mapping.loc[cat_mapping["raw_category"] == obj_label]
    #     if not cat_row.empty:
    #         cat_2_int = int(cat_row.iloc[0]["id"])
    #     else:
    #         raise ValueError(f"Category '{obj_label}' not found in category mapping for dataset '{dataset_name}'.")
        
    #     return cat_2_int


    # def select_random_category(self,
    #                            objects_in_scene: list[dict],
    #                            dataset_name: str):
    #     """
    #     Changing into a dictionary with label(key)-segments(value) pair, then selecting a random category
    #     """
    #     category_dict = defaultdict(list)

    #     for obj in objects_in_scene:
    #         cat_2_int = self.cat_2_int(dataset_name=dataset_name,
    #                        obj_label=obj["label"])
            
    #         category_dict[cat_2_int].extend(obj["segments"])

    #     random_cat = random.choice(list(category_dict.keys()))
    #     segments = category_dict[random_cat]

    #     return random_cat, segments

    # def generate_all_cat_mask(self,
    #                           segIndices: list,
    #                           objects_in_scene: list,
    #                           dataset_name: str,
    #                           scene_id: str):
    #     # TODO : Change this to separate binary masks!!
    #     mask = -np.ones(segIndices, dtype=int)
    #     categories = set()

    #     for obj in objects_in_scene:
    #         cat_2_int = self.cat_2_int(dataset_name=dataset_name,
    #                                    obj_label=obj["label"])
    #         categories.add(cat_2_int)

    #         obj_point_idx = np.isin(segIndices, obj["segments"])
    #         mask[obj_point_idx] = cat_2_int

    #     if np.any(np.isin(mask, -1)):
    #         raise Exception(f"{scene_id} in dataset {dataset_name} has unlabeled points")
        
    #     return categories, mask


            
    def __getitem__(self, index):
        # return super().__getitem__(index)
        # Dictionary with keys - 'scene_id', 'xyz', 'features', 'task', 'task_type', 'mask(labels)', 'category' (for specific), (+ @)
        dataset_name, scene_id = self.all_samples[index]

        # 1. Get dataset config to get file paths & Get info files
        dataset_paths = self.cfg[dataset_name]
        root_dir = Path(dataset_paths.root_dir)

        # seg_path = root_dir / scene_id / dataset_paths.segments_file_name
        # semseg_path = root_dir / scene_id / dataset_paths.semantics_file_name
        # cat_mapping = self.category_mapping_files[dataset_name]

        # seg_data = load_json(seg_path)
        # semseg_data = load_json(semseg_path)
        # segIndices = seg_data["segIndices"]
        # objects_in_scene = semseg_data["segGroups"]

        # if self.seg_type == "specific":
        #     # Select category (not object, since multiple objects can exist in a scene)
        #     random_category, segments = self.select_random_category(objects_in_scene, dataset_name)

        #     # Create category mask
        #     mask = np.isin(segIndices, segments)
        # elif self.seg_type == "all_category":
        #     all_cats, mask = self.generate_all_cat_mask(segIndices,
        #                                                 objects_in_scene,
        #                                                 dataset_name,
        #                                                 scene_id)
        # else:
        #     raise Exception("Other semantic segmentation besides specific category and all category are not implemented")

        # 2. Get pointcloud
        ply_path = root_dir / scene_id / dataset_paths.ply_name
        pcd = o3d.io.read_pointcloud(ply_path)
        xyz = np.asarray(pcd.points)
        color = np.asarray(pcd.colors)

        features = np.concatenate([xyz, color], axis=-1)
        features = self.preprocess_pcd(features)

        label_pcd = PlyData.read(ply_path)
        labels = np.asarray(label_pcd['vertex'].data['label'])

        if self.seg_type == "specific":
            if dataset_name == "ScanNet":
                random_category = random.choice(SCANNET_CLASS_IDS)
                mask = torch.from_numpy(labels == random_category).int()
            else:
                # Other dataset
                pass
        
        elif self.seg_type == "all_categories":
            if dataset_name == "ScanNet":
                valid_labels = np.isin(labels, SCANNET_CLASS_IDS)
                not_onehot_mask = np.full_like(labels, fill_value=-100)
                not_onehot_mask[valid_labels] = labels[valid_labels]
                all_cats = not_onehot_mask.unique()[1:]        # Exclude -100

                # Create a binary mask for each category in all_cats
                mask = (labels[None, :] == torch.from_numpy(all_cats)[:, None]).int()
                # mask = F.one_hot(mask, num_classes=len(SCANNET_CLASS_IDS))
                # TODO: mask should be in shape of (B, G, N)
                # Each (B, N) should be a binary mask
            else:
                # Other dataset
                pass


        # 3. Generate Data Dictionary
        # Store the indices instead of storing the raw string for converting torch tensors
        data_dict = {
            "dataset_idx": DATASETS.index(dataset_name),
            # "scene_id": SCENE_IDS[dataset_name].index(scene_id),      # Is this needed?
            "xyz": features[:,:3],
            "features": features,
            "task": TASKS.index("SemanticSegmentation"),
            "task_type": TASK_TYPE.index(self.seg_type),
            "mask": mask
        }
        if self.seg_type == "specific":
            data_dict["category"] = random_category
        elif self.seg_type == "all_category":
            data_dict["category"] = all_cats
            data_dict["num_category"] = len(all_cats)

        return data_dict

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


def default_collate_fn():
    pass


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
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            shuffle=True
        )

        val_dataset = SemanticSegmentationDataset(
            split="val",
            segmentation_type=task_type,
            dataset_cfg_path=cfg.dataset_config_path,
            instructions_cfg_path=cfg.instructions_config_path
        )
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=cfg.val.batch_size,
            num_workers=cfg.val.num_workers,
            pin_memory=cfg.val.pin_memory,
            shuffle=False
        )

        test_dataset = SemanticSegmentationDataset(
            split="test",
            segmentation_type=task_type,
            dataset_cfg_path=cfg.dataset_config_path,
            instructions_cfg_path=cfg.instructions_config_path
        )
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.test.num_workers,
            pin_memory=cfg.test.pin_memory,
            shuffle=False
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