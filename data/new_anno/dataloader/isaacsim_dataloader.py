import os
from pathlib import Path
import pickle
from collections import defaultdict
from glob import glob
import re

import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import DBSCAN

from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Union

from .objects import pcObject
from spatial_relations.support import is_suppported
from utils import check_spell


class CustomTabletopDataLoader:
    """
    Loader class for point cloud scenes
    Loading point cloud scenes and per-point labels from custom tabletop dataset
    (Custom Dataset is generated via IsaacSim)

    """
    def __init__(self, config: Union[DictConfig, ListConfig]):
        """
        Initialize the dataloader

        Args:
            config (dict) : Configurations for dataloader and each scene
        """

        self.verbose = config.verbose
        self.dry_run = config.dry_run

        self.dataset_config = config.Datasets["IsaacSimTabletop"]
        self.base_dir = Path(self.dataset_config.base_dir).resolve()
        self.scene_dir = self.dataset_config.scene_dir
        self.ply_filename = self.dataset_config.ply_name
        
        self.log_file = self.dataset_config.log_file
        self.table_height = self.dataset_config.table_height
        self.eps = self.dataset_config.eps

        # self.supporter_objects = ["table"]

        if self.verbose:
            print("Start loading data of IsaacSimTabletop..")


        # Load object labels and object centers from log file
        # TODO: LOAD log file and figure out the structure of the log file
        log_file_path = os.path.join(self.base_dir, self.log_file)
        self.infos_df = pd.read_csv(log_file_path)

    def get_objects(self, ply_name: str):
        scene_id = int(re.findall(r'\d+', ply_name)[0])

        objects_df = self.infos_df[self.infos_df["scene_index"] == scene_id]
        scene_pcd = o3d.io.read_point_cloud(os.path.join(self.base_dir /'scenes' / ply_name))
        scene_points = np.asarray(scene_pcd.points)
        scene_rgb = np.asarray(scene_pcd.colors)

        scene_objects = defaultdict(list)

        # ==== 1. Table Object ================================
        table = pcObject()
        is_table = scene_points[:,2] < self.table_height + self.eps     # For tolerance

        table.label = "table"
        table.points = scene_points[is_table]
        table.idx_in_scene = is_table
        table.color = scene_rgb[is_table]
        # table.check_label_validity(is_custom=True)

        scene_objects["supporting"].append(table)

        if self.verbose:
            print("Finished Loading Table!")


        # ==== 2. Other objects supported by the table ========
        obj_points = scene_points[~is_table]

        dbscan = DBSCAN(eps=0.1, min_samples=50)
        labels = dbscan.fit_predict(obj_points)

        # Compute cluster centers (centroids)
        cluster_centers = []
        for cluster_id in np.unique(labels):
            if cluster_id != -1:
                cluster_points = obj_points[labels == cluster_id]
                centroid = cluster_points.mean(axis=0)
                cluster_centers.append(centroid[:2])
        cluster_centers = np.stack(cluster_centers, axis=0)

        for i, obj in objects_df.iterrows():
            object_in_scene = pcObject()

            object_in_scene.label = obj["name"]
            object_in_scene.center = (obj['x'], obj['y'])

            label_id = np.linalg.norm(cluster_centers - object_in_scene.center, axis=1).argmin()
            object_in_scene.points = obj_points[labels == label_id]

            is_object_in_scene = np.zeros(scene_points.shape[0], dtype=bool)
            is_object_in_scene[~is_table] = labels == label_id
            object_in_scene.idx_in_scene = is_object_in_scene

            object_in_scene.color = scene_rgb[is_object_in_scene]
            # object_in_scene.check_label_validity(is_custom=True)

            scene_objects["supported_by"].append(object_in_scene)

        if self.verbose:
            print("Finished Loading objects on the table!")
    
        return scene_objects
    
    def create_hierarchy(self, all_objects: dict):
        """
        Create object hierarchy for all objects in the scene
        1. Parent objects are the supporting object (table)
        2. Children objects are the objects suppported by the supporting object
        """

        n_relations = 0

        tables = all_objects["supporting"]

        if len(tables) != 0:
            other_objects = all_objects["supported_by"]

            for table in tables:
                for obj in other_objects:
                        table.add_child(obj)
                        obj.parent = table
                        n_relations += 1
        
        return n_relations > 0


    def get_objects_for_all_scene(self):
        """
        Get all object hierarchy for all scene
        """

        data_dict = {}

        scenes = [name for name in os.listdir(self.base_dir / self.scene_dir)]
        
        for ply_name in scenes:
            if self.verbose:
                print(f"Processing {ply_name}...")
            scene_objects = self.get_objects(ply_name)
            is_valid_scene = self.create_hierarchy(scene_objects)

            if is_valid_scene:
                scene_name = ply_name.rstrip(self.ply_filename)
                data_dict[scene_name] = scene_objects

            if self.dry_run:
                break

        return data_dict