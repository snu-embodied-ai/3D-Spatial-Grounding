import os, sys
from pathlib import Path
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from collections import defaultdict
from itertools import chain

import numpy as np
from omegaconf import OmegaConf

from dataloader.dataloader import PointCloudDataLoader
from dataloader.objects import pcObject
from detect_freespace_smooth import *
from spatial_relations.relationships import UNARY, PAIRWISE
from utils import parse_args, find_duplicates


PREFIX = ["the free space", "the feasible region"]

def generate_unary(obj: pcObject,
                   surface: pcObject,
                   free_space_on_surface: np.ndarray[bool],
                   free_space_scores: np.ndarray[float],
                   verbose: bool = True):

    surface_identifier = f"on the {surface.label}"

    # 1. Generate free space mask compatible to the whole scene
    free_space_mask = np.zeros_like(surface.idx_in_scene, dtype=bool)
    scores = np.zeros_like(surface.idx_in_scene, dtype=float)
    
    free_space_mask[surface.idx_in_scene] = free_space_on_surface
    scores[surface.idx_in_scene] = free_space_scores

    # 2. Create dictionary to save the final result
    spatial_descriptions = []

    # 3. Generate description for all cases
    for rel_type, relations in UNARY.items():
        # 4. Case 1: no perspective -> no additional filtering needed for final free space mask
        if rel_type == "no perspective":
            # TODO : Add a description refinement (beautify) implementation
            prefix = random.choice(PREFIX)
            rel = random.choice(relations)
            description = f"{prefix} {surface_identifier} {rel} the {obj.final_label}"

            if verbose:
                print("Generated Unary Description: ", description)

            spatial_descriptions.append({
                "description": description,
                "free_space_mask": free_space_mask,
                "free_space_scores": scores
            })

        # 5. Case 2: perspective -> additional filtering needed for final free space mask (TODO)
        elif rel_type == "perspective":
            pass        # Should think about this relation type


    return spatial_descriptions

def generate_binary(obj1: pcObject,
                    obj2: pcObject,
                    surface: pcObject,
                    free_space_on_surface: np.ndarray[bool],
                    free_space_scores: np.ndarray[float],
                    verbose: bool = True):

    surface_identifier = f"on the {surface.label}"

    # 1. Generate free space mask compatible to the whole scene
    free_space_mask = np.zeros_like(surface.idx_in_scene, dtype=bool)
    scores = np.zeros_like(surface.idx_in_scene, dtype=float)

    free_space_mask[surface.idx_in_scene] = free_space_on_surface
    scores[surface.idx_in_scene] = free_space_scores

    # 2. Create dictionary to save the final result
    spatial_descriptions = []

    # 3. Generate description for all cases
    for rel_type, relations in PAIRWISE.items():
        # 4. Case 1: no perspective -> no additional filtering needed for final free space mask
        if rel_type == "no perspective":
            # TODO : Add a description refinement (beautify) implementation
            prefix = random.choice(PREFIX)
            rel = random.choice(relations)
            description = f"{prefix} {surface_identifier} {rel} the {obj1.final_label} and {obj2.final_label}"

            if verbose:
                print("Generated Binary Description: ", description)

            spatial_descriptions.append({
                "description": description,
                "free_space_mask": free_space_mask,
                "free_space_scores": scores
            })

        # 5. Case 2: perspective -> additional filtering needed for final free space mask (TODO)
        elif rel_type == "perspective":
            pass        # Should think about this relation type


    return spatial_descriptions

def generate_annotation_surface(surface: pcObject,
                                surface_descriptions: defaultdict[list],
                                surface_threshold: float,
                                binary_threshold_scale: float,
                                verbose: bool):
    
    for obj in surface.children:
        obj.check_label_validity()
        
    # 1. Define all free space above the surface
    all_free_space = define_freespace(surface)

    # 1-1. Find duplicate objects and divide them
    divided_objects, wall_objects = find_duplicates(surface.children)
    if verbose:
        print(f"Objects grouped by same label on {surface.label}: ", divided_objects)
        
    all_unary_descriptions = []
    for i, (obj_label, objects) in enumerate(divided_objects.items()):
        unary_free_space = np.zeros_like(all_free_space)
        unary_free_space_scores = np.zeros_like(all_free_space, dtype=float)

        for obj in objects:
            # 1. Define free space for unary relations
            single_free_space, free_space_scores = freespace_near_single(obj=obj,
                                                                         surface=surface,
                                                                         walls=wall_objects,
                                                                         free_space_mask=all_free_space,
                                                                         threshold=surface_threshold)
            unary_free_space = np.logical_or(unary_free_space, single_free_space)
            unary_free_space_scores += free_space_scores
        
        unary_free_space_scores /= len(objects)

        # 2. Generate unary spatial relation
        unary_descriptions = generate_unary(obj=objects[0],
                                            surface=surface,
                                            free_space_on_surface=unary_free_space,
                                            free_space_scores=unary_free_space_scores)
        all_unary_descriptions.extend(unary_descriptions)
        
    surface_descriptions["unary"].extend(all_unary_descriptions)

    valid_objects = list(chain.from_iterable(divided_objects.values()))
            
    if len(valid_objects) > 1:
        # 3. Compute pairwise distances of objects on the same surface
        pair_ids_with_dist = get_object_pairs(all_objects=valid_objects,
                                              walls=wall_objects,
                                              binary_threshold=binary_threshold_scale * surface_threshold)
            
        all_binary_descriptions = []
        for id_1, id_2, dist in pair_ids_with_dist:
            # 0. Check if the object is valid for annotation
            obj1 = valid_objects[id_1]
            obj2 = valid_objects[id_2]

            if obj1.final_label is None or obj2.final_label is None:
                continue

            # 4. Define free space between those closest object pairs
            binary_free_space, free_space_scores = freespace_near_pair(obj1=obj1,
                                                                       obj2=obj2,
                                                                       surface=surface,
                                                                       pair_dist=dist,
                                                                       free_space_mask=all_free_space)
                
            # 5. Generate binary relation
            binary_descriptions = generate_binary(obj1=obj1,
                                                  obj2=obj2,
                                                  surface=surface,
                                                  free_space_on_surface=binary_free_space,
                                                  free_space_scores=free_space_scores)
            all_binary_descriptions.extend(binary_descriptions)

        surface_descriptions["binary"].extend(all_binary_descriptions)

    return surface_descriptions

def generate_annotations(all_objects: dict,
                         floor_threshold: float,
                         surface_threshold: float,
                         binary_threshold_scale: float,
                         verbose: bool = True):

    # ====== FLOOR DESCRIPTION GENERATION ==============================
    floor_objs = all_objects["floor"]
    floor_descriptions = defaultdict(list)

    for floor in floor_objs:
        if verbose:
            print("FLOOR label: ", floor.label)

        floor_descriptions = generate_annotation_surface(surface=floor,
                                                         surface_descriptions=floor_descriptions,
                                                         surface_threshold=floor_threshold,
                                                         binary_threshold_scale=binary_threshold_scale,
                                                         verbose=verbose)

    # ====== SURFACE DESCRIPTION GENERATION ==============================
    surface_objs = all_objects["supporting"]
    surface_descriptions = defaultdict(list)

    for surface in surface_objs:
        if verbose:
            print("SURFACE label: ", surface.label)
        
        surface_descriptions = generate_annotation_surface(surface=surface,
                                                           surface_descriptions=surface_descriptions,
                                                           surface_threshold=surface_threshold,
                                                           binary_threshold_scale=binary_threshold_scale,
                                                           verbose=verbose)

    return {
        "floor" : floor_descriptions,
        "non-floor" : surface_descriptions
    }


# Helper function must be top-level (not nested) for pickling
def process_scene(args):
    scene_name, scene_objects, config_dict = args

    result = generate_annotations(
        scene_objects,
        floor_threshold=config_dict['floor_threshold'],
        surface_threshold=config_dict['surface_threshold'],
        binary_threshold_scale=config_dict['binary_threshold_scale']
    )

    return scene_name, result


def run(config: OmegaConf, 
        num_workers: int):
    
    if config.verbose:
        print("Start running annotation process...")

    all_annotations = {}

    for dataset_name in config.Datasets:
        dataloader = PointCloudDataLoader(config=config,
                                          dataset_name=dataset_name)
        
        scene_dict = dataloader.get_objects_for_all_scene()
        scene_annos = dict()

        # Prepare configs as a simple dict (for pickling)
        config_dict = {
            'floor_threshold': config.floor_threshold,
            'surface_threshold': config.surface_threshold,
            'binary_threshold_scale': config.binary_threshold_scale
        }

        # Pack args for multiprocessing
        scene_args = [
            (scene_name, scene_objects, config_dict)
            for scene_name, scene_objects in scene_dict.items()
        ]

        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_scene, args) for args in scene_args]
            for future in as_completed(futures):
                scene_name, result = future.result()
                scene_annos[scene_name] = result

        # for scene_name, scene_objects in scene_dict.items():
        #     if config.verbose:
        #         print(f"Processing scene {scene_name} of dataset {dataset_name}")
        #     scene_annos[scene_name] = generate_annotations(scene_objects,
        #                                                    floor_threshold=config.floor_threshold,
        #                                                    surface_threshold=config.surface_threshold,
        #                                                    binary_threshold_scale=config.binary_threshold_scale,
        #                                                    verbose=config.verbose)
        
        if config.verbose:
            print(f"Done processing {dataset_name}...\n")
        all_annotations[dataset_name] = scene_annos

        if config.dry_run:
            break

    # Write the bytes to a file
    os.makedirs(config.output_dir, exist_ok=True)
    anno_file_path = os.path.join(config.output_dir, "output_annotation.pkl")

    if config.verbose:
        print(f"Saving annotations to {anno_file_path}...")

    with open(anno_file_path, "wb") as f:
        pickle.dump(all_annotations, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    cur_dir = Path(__file__).resolve().parent
    child_dirs = [p for p in cur_dir.iterdir() if p.is_dir()]
    sys.path.extend(child_dirs)

    args = parse_args()
    config = OmegaConf.load(args.config_path)

    run(config=config, num_workers=args.num_workers)