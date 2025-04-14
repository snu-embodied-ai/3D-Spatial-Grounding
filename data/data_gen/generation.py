import numpy as np
import open3d as o3d
import os
import argparse
import random
import json
import pandas as pd
from tqdm import tqdm

import yaml

from util import GraspNetObject, Surface, Description, RELATIONS_DICT


def main(args):

    """
    The DATASET GENERATION STEPS
비밀번
    **Hyperparameters**
    - `num_objects` : Number of objects to include in the whole scene
    - `include_duplicates` : Whether to include duplicate objects for the main objects \\
                             The main objects are the objects used to create the description

    1. Create the supporting surface
    2. Select a relation to include in the text description
    3. Create the main objects, then locate them according to the relations
    4. (VERIFICATION) Check if the relations are valid after locating the objects
    5. Create distractor objects
    6. Randomly place one distractor object in the free space, then check if the relations are still valid
    7. Iterate this for all distractor objects. If a pre-created distractor cannot fit into the scene, \\
       discard it and create another distractor object and repeat the *distractor placement* process
    8. Create answer regions, which are the correct region according to the text description
    """

    # 1. Create the surface
    

    # 2. Select the relation for the text description

    # TODO: Add COMPLICATED RELATIONS
    # 1. Multiple relations : Right to the apple and behind the banana
    # 2. More than one region : In front of the banana (multiple bananas in the scene)
    # 3. View-dependent relations : left, right, in front of, behind, etc.
    # 4. Distance/Degree (Exact number)
    
    obj_labels = pd.read_csv(os.path.join(config["graspnet_dir"], "obj_labels.csv"))

    all_descriptions = []
    labels_to_save = pd.DataFrame()

    idx = 0
    for n in tqdm(range(config["scene"]["num_samples"]), desc="Creating scenes - next to, besides, between"):
        for i, (key, relations_list) in enumerate(RELATIONS_DICT.items()):
            for relation in relations_list:
                idx += 1
                # Create description
                description = Description(relation, key, obj_labels, 
                                          config["graspnet_dir"],
                                          config["scene"]["between_threshold"],
                                          config["scene"]["besides_threshold"],
                                          config["scene"]["describe_freespace"])

                placed = False
                surface = None
                while not placed:
                    surface = Surface(width=config["scene"]["surface_width"], 
                                      height=config["scene"]["surface_height"], 
                                      thickness=config["scene"]["surface_thickness"],
                                      grid_size=config["scene"]["surface_grid_size"], 
                                      texture="wood", paint_color=True,
                                      density=config["scene"]["surface_density"])
                    surface.create_mesh()
                    description.reset()
                    placed = description.place_main_objects(surface)

                print(f"Done placing main objects - <{description.get_description()}>")

                # TODO: Add distractor objects
                placed_distractors = description.place_all_distractors(config["scene"]["num_objects"], surface)
                if not placed_distractors:
                    continue

                all_descriptions.append(description)
                print(f"Done creating <{description.get_description()}>")

                labels_to_save = description.save_scene(surface, config["save_dir"], idx, labels_to_save)

    labels_path = os.path.join(config["save_dir"], "labels.csv")
    labels_to_save.to_csv(labels_path)


    # TEST
    # main_objects = all_descriptions[-1].get_main_objects()
    # distractors = all_descriptions[-1].get_distractors()
    # for_vis_main = [obj.get_mesh() for obj in main_objects+distractors]
    # label = all_descriptions[-1].create_heatmap_label(surface)
    # surface.draw_heatmap(label)
    # for_vis_main.append(surface.get_mesh())
    
    
    # if config["connection"] == "remote":
    #     ev = o3d.visualization.ExternalVisualizer(timeout=2000000)
    #     draw = ev.draw
    #     draw(for_vis_main)
    # elif config["connection"] == "local":
    #     o3d.visualization.draw(for_vis_main, title=all_descriptions[-1].get_description())
    # elif config["connection"] == "web":
    #     o3d.visualization.webrtc_server.enable_webrtc()
    #     o3d.visualization.draw(for_vis_main)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="/media/jslee/219db482-9799-4615-841a-d8f46e93e50c/home/kykwon/SpatialUnderstanding/data_gen/generation.yaml", type=str)

    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
