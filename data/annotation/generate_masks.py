from pathlib import Path
import argparse
import glob
import pickle
from collections import defaultdict
from typing import Dict, List

import msgspec, json
import numpy as np
import open3d as o3d
import pandas as pd

from freespace import get_all_freespace
from compute_relation import compute_free_space
from object_loader import load_and_group_objects

def load_gpt_labels(file_path: Path):
    
    descriptions = []
    with open(file_path, 'r') as j:
        for line in j:
            data = msgspec.json.decode(line)
            descriptions.append(data)

    return descriptions

def refine_obj_names(id_to_label: dict):
    refined = dict()

    for id, name in id_to_label.items():
        if name.startswith("cabinet"):
            splits = name.split("_")
            open_level = splits[2]

            fixed_name = f"drawer with {open_level} level open"
        else:
            fixed_name = name.replace("_", " ")

        refined[fixed_name] = id

    return refined

def load_data(scene_id: int,
              root_dir: Path):
    
    # 1. Get paths for each data
    pose_paths = [root_dir / "pose" / f"pose_{cam_id}.npy" for cam_id in [0, 1, 2]]
    pcd_path = root_dir / "global_clouds" / f"pcd_{scene_id}.ply"
    segments_path = root_dir / "global_clouds" / f"segments_{scene_id}.npy"
    # semantic_info_path = root_dir / "semantic" / f"semantic_idToLabel_frame_{scene_id}_0.json"
    # instance_info_path = root_dir / "instance" / f"instance_idToLabel_frame_{scene_id}_0.json"

    log_path = root_dir / "log"

    # 2. Load camera poses
    poses = np.concatenate([np.load(path) for path in pose_paths], axis=0)

    # 3. Load pointcloud data
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)
    features = np.asarray(pcd.colors)
    pcd_feats = np.concatenate([points, features], axis=-1)

    # 4. Load segmentation data
    segments = np.load(segments_path)

    # 5. Load object informations in the scene
    df = pd.read_csv(log_path)
    id_to_label = df[df['scene_index'] == scene_id].to_dict()["name"]
    label_to_id = refine_obj_names(id_to_label)
    label_to_id["table"] = len(label_to_id)

    return {
        "pcd_feats": pcd_feats,
        "poses": poses,
        "segments": segments,
        "label_to_id": label_to_id
    }

def main(args):
    root_dir = args.data_dir / args.scene_type
    gpt_labels_path = glob.glob(str(root_dir / "*.jsonl"))[0]
    descriptions = load_gpt_labels(gpt_labels_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "freespace_label.pkl"

    final_labels = defaultdict(list)
    
    for scene_anno in descriptions:
        scene_id = int(scene_anno["scene_id"])
        cameras = scene_anno["cameras"]

        data = load_data(scene_id, root_dir)

        for img_desc in cameras:
            cam_id = int(img_desc["camera_id"])
            # all_objects = img_desc["all_objects"]

            objects_dict, instances = load_and_group_objects(data)

            all_freespace = get_all_freespace(instances)

            for result in img_desc["results"]:
                support = result["supporting_object"]

                for i, rel in enumerate(result["targets_related"]):
                    relation = rel["relation"]
                    targets = rel["targets"]

                    # Find appropriate targets which are in the given relation & on the given supporting surface and compute free space
                    freespace, scores = compute_free_space(relation, support, targets, objects_dict)
                    if freespace is None:
                        continue

                    if i == 0:
                        final_freespace = freespace
                        final_scores = scores
                    else:
                        final_freespace = np.logical_and(final_freespace, freespace)
                        final_scores *= scores

                final_labels[scene_id].append({
                    "freespace_mask": final_freespace,
                    "scores": final_scores,
                    "camera_id": cam_id,
                    "description": result["description"],
                })
                        
    with open(output_path, 'wb') as f:
        pickle.dump(final_labels, f, protocol=pickle.HIGHEST_PROTOCOL)              



# def main(args):
#     root_dir = args.data_dir / args.scene_type
#     gpt_labels_path = glob.glob(str(root_dir / "*.jsonl"))[0]
#     descriptions = load_gpt_labels(gpt_labels_path)

#     output_path = args.output_dir / "freespace_label.pkl"

#     final_labels = defaultdict(List)
#     for img_desc in descriptions:
#         scene_id = int(img_desc["scene_id"])
#         cam_id = int(img_desc["camera_id"])
#         all_objects = img_desc["all_objects"]

#         data = load_data(scene_id, root_dir, scene_id)

#         objects_dict, instances = load_and_group_objects(data)

#         all_freespace = get_all_freespace(instances)

#         # Extract all object point features
#         all_objects, supporters = load_all_objects(all_objects, data)

#         # Get all free space on the scene (save to reduce massive computation)
#         all_freespace = get_all_freespace(supporters)
         
#         for result in img_desc["results"]:
#             target_objs = result["target_objects"]
#             support_obj = result["supporting_object"]
#             relation = result["relation"]
#             desc = result["description"]

#             # Get target/support obj pcd from all_objects
#             targets = [all_objects[obj] for obj in target_objs]
#             supports = all_objects[support_obj]

#             # Find appropriate targets-supports that are in contact
#             all_groups = []
#             for support in supports:
#                 group = {
#                     "supporter": support,
#                     "objects_on": defaultdict(list)
#                 }
#                 is_valid = True

#                 for tgt in targets:
#                     for obj in tgt:
#                         if obj.supporting_object == support and obj in support.objects_on_surface[obj.name]:
#                             group["objects_on"][obj.name].append(obj)
                        
#                     if len(group["objects_on"][obj.name]) == 0:
#                         is_valid = False
#                         break
                
#                 if is_valid:
#                     all_groups.append(group)

#             # Compute free space and assign scores
#             freespace, scores = compute_free_space(relation, all_groups, all_freespace)

#             # SAVE single result
#             final_labels[scene_id].append({
#                 "freespace_mask": freespace,
#                 "scores": scores,
#                 "camera_id": cam_id,
#                 "description": desc,
#             })

#         print(f"Done labeling pointcloud of scene {scene_id} (camera {cam_id})")

#     print("DONE LABELLING ALL THE DESCRIPTIONS!")
    
#     with open(output_path, 'wb') as f:
#         pickle.dump(final_labels, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene_type", type=str, required=True)
    parser.add_argument("--data_dir", type=Path, default="tabletop_dataset")
    parser.add_argument("--output_dir", type=Path, default="output")

    args = parser.parse_args()

    main(args)