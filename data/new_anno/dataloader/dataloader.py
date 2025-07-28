import os
import pickle
from collections import defaultdict
from glob import glob
import re

import numpy as np
import torch
import open3d as o3d
import pandas as pd

import msgspec
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Union

from .objects import pcObject
from spatial_relations.support import is_suppported
from utils import check_spell

class PointCloudDataLoader:
    """
    Loader class for point cloud scenes
    Loading point cloud scenes and per-point labels from original 3D scan datasets

    Supports:
    - Matterport3D
    - Scannet
    - 3RScan
    - (Work In Progress) S3DIS, Structured3D, HM3D, and so on. (, ARKitScenes)

    """

    def __init__(self, config: Union[DictConfig, ListConfig],
                 dataset_name: str):
        """
        Initialize the dataloader

        Args:
            config (dict) : Configurations for dataloader and each scene
        """

        self.verbose = config.verbose
        self.dry_run = config.dry_run

        self.dataset_name = dataset_name
        self.dataset_config = config.Datasets[dataset_name]
        self.base_dir = self.dataset_config.base_dir
        self.suffix = self.dataset_config.after_scene_name
        self.ply_filename = self.dataset_config.ply_name
        self.semseg_filename = self.dataset_config.semantics_file_name
        self.seg_filename = self.dataset_config.segments_file_name
        self.category_mapping_file = self.dataset_config.category_mapping_file

        self.supporter_objects = config.supporter_objects

        if self.verbose:
            print(f"Start loading data of {dataset_name}...")


        # Load label mapping csv(tsv) files of each dataset 
        # To convert label strings into integers and save per-point labels into a numpy array

        # ==== 1. Get the path of the mapping csv(tsv) ====================
        file_path = os.path.join(self.base_dir, self.category_mapping_file)

        # ==== 2. Load the mapping csv(tsv) into a pandas DataFrame =======
        separator = "\t" if file_path.endswith(".tsv") else ","
        df = pd.read_csv(file_path, sep=separator)

        if dataset_name == "3RScan":
            major_cats = ["NYU40 Mapping", "RIO27 Mapping"]
        elif dataset_name == "Matterport3D" or dataset_name == "Scannet":
            major_cats = ["nyu40class", "mpcat40"]

        id_col, label_col = df.columns[:2]
        self.mapping_df = df.set_index(label_col)[[id_col, *major_cats]]

    def get_objects(self, 
                   scene_name: str):
        scene_dir = os.path.join(self.base_dir, scene_name, self.suffix)

        # For Matterport3D, load the regions separately. - not the whole scene directly

        ply_paths = sorted(glob(os.path.join(scene_dir, f"*{self.ply_filename}")))
        seg_paths = sorted(glob(os.path.join(scene_dir, f"*{self.seg_filename}")))
        sem_paths = sorted(glob(os.path.join(scene_dir, f"*{self.semseg_filename}")))

        scene_data = []
        for i, (ply_file, seg_file, sem_file) in enumerate(zip(ply_paths, seg_paths, sem_paths)):
            with open(seg_file) as f:
                json_content = f.read()
                json_content = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', json_content)
                seg_data = msgspec.json.decode(json_content)
            with open(sem_file) as j:
                if self.verbose:
                    print(sem_file)
                json_content = j.read()
                json_content = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', json_content)
                sem_data = msgspec.json.decode(json_content)

            pcd = o3d.io.read_point_cloud(ply_file)
            points = np.asarray(pcd.points)
            rgb = np.asarray(pcd.colors)

            # 3. Load segIndices
            seg_indices = seg_data['segIndices']
            assert len(seg_indices) == len(points), "Mismatch between points and seg indices."

            scene_objects = defaultdict(list)
            # scene_objects["all_points"] = points

            floor = pcObject()
            floor_exists = False
            floor_idx_in_scene = np.zeros_like(seg_indices)

            for group in sem_data['segGroups']:
                try:
                    label_id, *major_cats = self.mapping_df.loc[group['label']]
                except:
                    if group['label'] == "workbench":
                        label_id, *major_cats = self.mapping_df.loc["work bench"]
                    elif group['label'] == "roof or floor / other room":
                        label_id, *major_cats = self.mapping_df.loc["floor / other room"]
                    else:
                        modified_label = check_spell(group['label'])
                        if self.verbose:
                            print(f"\n==== Typo found :{group['label']}, Converted to {modified_label} ====\n")
                        label_id, *major_cats = self.mapping_df.loc[modified_label]
                        group['label'] = modified_label
                object_in_scene = pcObject()

                point_idx = np.isin(np.array(seg_indices), np.array(group['segments']))

                if "ceiling" in major_cats:
                    continue

                if "floor" in major_cats:
                    floor_exists = True
                    floor_idx_in_scene = np.logical_or(floor_idx_in_scene, point_idx)
                else:
                    if any(cat in self.supporter_objects for cat in major_cats):
                        obj_type = "supporting"
                    else:
                        obj_type = "supported_by"

                    object_in_scene.idx_in_scene = point_idx
                    object_in_scene.points = points[point_idx]
                    object_in_scene.color = rgb[point_idx]

                    object_in_scene.label = group['label']
                    object_in_scene.label_id = label_id
                    object_in_scene.obj_type = obj_type
                    object_in_scene.major_category = major_cats

                    scene_objects[obj_type].append(object_in_scene)

            if floor_exists:
                floor.obj_type = "floor"
                floor.label = "floor"
                floor.label_id, *floor.major_category = self.mapping_df.loc["floor"]
                floor.idx_in_scene = floor_idx_in_scene
                floor.points = points[floor_idx_in_scene]
                floor.color = rgb[floor_idx_in_scene]
                scene_objects[floor.obj_type].append(floor)

            scene_data.append(scene_objects)
            # TODO : NO need for per-point label
            # JUST directly find the supporter/non-supporter
            # Create containers for supporters and non-supporters eachs
            # and find supp-nonsupp pair which are in contact
            # Create tree-style container to save supporters as a parent and in-contact non-supporters as children
            # for in contact no-supporter objects,
            # Find free space for unary relation
            # And compute spatial relations with other same condition non-supporters
            # Then find free space

        return scene_data
    
    def create_hierarchy(self,
                         all_objects: dict):
        """
        Create object hierarchy for all objects in the scene
        1. Parent objects are the supporting object (floor, table, etc.)
        2. Children objects are the objects suppported by the supporting object
        """

        n_relations = 0

        # 1. Get floor and set it as a parent object
        floor_objects = all_objects["floor"]

        if len(floor_objects) == 0:
            # SKIP current scene if no floor objects are available
            return False
        else:
            obj_supporting = all_objects["supporting"]
            obj_supported_by = all_objects["supported_by"]
            other_objects = obj_supporting + obj_supported_by

            for floor in floor_objects:
                for obj in other_objects:
                    # Compute "in-contact" relation
                    # Then append to children of parent object
                    # by `add_children()`
                    in_contact_rel = is_suppported(obj, floor)

                    if not in_contact_rel:      # not in contact
                        continue
                    elif in_contact_rel == "inside_express":
                        continue
                    else:       # "support_express", "embed_express"
                        floor.add_child(obj)
                        obj.parent = floor
                        n_relations += 1
                    

        # 2. Get supporting objects (table, desk, counter) and set them as parent objects
        supporting_objects = all_objects["supporting"]

        if len(supporting_objects) != 0:
            other_objects = all_objects["supported_by"]

            for supporting in supporting_objects:
                for obj in other_objects:
                    # Compute "in-contact" relation
                    # Then append to children of parent object
                    # by `add_children()`
                    in_contact_rel = is_suppported(obj, supporting)

                    if not in_contact_rel:      # not in contact
                        continue
                    elif in_contact_rel == "inside_express":
                        continue
                    else:       # "support_express", "embed_express"
                        supporting.add_child(obj)
                        obj.parent = supporting
                        n_relations += 1

        # 3. Return False if no relations are created
        return bool(n_relations)

    def get_objects_for_all_scene(self):
        """
        Get all object hierarchy for all scene
        """
        data_dict = {}

        scenes = [name for name in os.listdir(self.base_dir) \
                  if os.path.isdir(os.path.join(self.base_dir, name))]
        
        for scene_name in scenes:
            scene_obj_data = self.get_objects(scene_name)

            # ==== Create hierarchy -> supporter - non_supporters
            is_valid_scene = []
            for scene_objects in scene_obj_data:
                is_valid_scene.append(self.create_hierarchy(scene_objects))

            for i, is_valid in enumerate(is_valid_scene):
                if is_valid:
                    if self.dataset_name == "Matterport3D":
                        data_dict[f"{scene_name}_{i}"] = scene_obj_data[i]
                    else:
                        data_dict[scene_name] = scene_obj_data[i]

            if self.dry_run:
                break

        return data_dict
