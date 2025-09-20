from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import msgspec, json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from freespace import get_single_obj_freespace, get_pair_freespace
from utils import apply_gaussian_smoothing, get_subplanes_vertical, match_neighboring_pairs

def compute_next_to(support_and_tgts: Dict,):
    """
    Generating a mask of free space which are located next to the target objects

    Parameters
    ---
    support_and_tgts: Dict
        Dict of supporting object and target objects
    """
    # 1. Suppport object
    support = support_and_tgts["supporter"]
    objects_on = support_and_tgts["objects_on"]

    for i, (name, tgt_insts) in enumerate(objects_on.items()):
        for j, instance in enumerate(tgt_insts):
            single_free_space, single_scores = get_single_obj_freespace(instance, support)

            if j == 0:
                inst_free_space = single_free_space
                inst_scores = single_scores
            else:
                inst_free_space = inst_free_space | single_free_space
                inst_scores = (inst_scores + single_scores) - (inst_scores * single_scores)
        inst_scores /= inst_scores.max()

        if i == 0:
            free_space = inst_free_space
            scores = inst_scores
        else:
            free_space = free_space & inst_free_space
            scores *= inst_scores

    # 3. Assign scores to free space based on distance
    scores[~free_space] = 0
    scores /= scores.max()          # Divide by max value to set the max score to 1

    return free_space, scores


def compute_vertical(support_and_tgts: Dict,
                     direction: str = "below",
                     min_num_points: int = 200,
                     threshold: float = 0.2,
                     slope: float = 9.0):
    """
    Generating a mask of free space which are located next to the target objects

    Parameters
    ---
    support_and_tgts: list
        List of supporting object and target objects
    direction: str
        Relation string showing the vertical direction. Only "below", "above", "on" relations are allowed.
    threshold: float
        Maximum distance for 'next to' relation
    peak_point: float
    """

    assert direction in ["below", "above", "on"], ValueError("Only \"below\", \"on\" and \"above\" directions are allowed.")

    # 1. Suppport object
    support = support_and_tgts["supporter"]
    objects_on = support_and_tgts.get("objects_on", None)

    # "on" relations
    if objects_on is None:
        # A. Get subplane points and assign scores to each.
        # Assgin scores to subplanes that have enough number of points
        plane_pts = support.inst_feats[:,:3][support.surface_mask]

        mid_point = plane_pts.mean(axis=0)
        dists = np.linalg.norm(plane_pts - mid_point, axis=1)
        
        final_mask = np.zeros_like(support.inst_mask)
        final_mask[support.inst_mask][support.surface_mask] = dists <= threshold
        valid_dists = dists[dists <= threshold]

        single_scores = apply_gaussian_smoothing(mean=0, std=threshold / slope,
                                                    valid_dists=valid_dists,
                                                    final_mask=final_mask)
        
        scores = single_scores / single_scores.max()
        free_space = final_mask

        # for i, mask in enumerate(support.surfaces):
        #     plane_pts = support.inst_feats[:,:3][mask]

        #     if plane_pts.shape[0] < min_num_points:
        #         continue
            
        #     mid_point = plane_pts.mean(axis=0)
        #     dists = np.linalg.norm(plane_pts - mid_point, axis=1)
            
        #     final_mask = np.zeros_like(support.inst_mask)
        #     final_mask[support.inst_mask][mask] = dists <= threshold
        #     valid_dists = dists[dists <= threshold]

        #     single_scores = apply_gaussian_smoothing(mean=0, std=threshold / slope,
        #                                              valid_dists=valid_dists,
        #                                              final_mask=final_mask)
            
        #     if i == 0:
        #         free_space = final_mask
        #         scores = single_scores
        #     else:
        #         free_space = free_space | single_free_space
        #         scores = (scores + single_scores) - (scores * single_scores)

        # scores /= scores.max()
    else:
        for i, (name, tgt_insts) in enumerate(objects_on.items()):
            for j, instance in enumerate(tgt_insts):
                # non_contact_plane_masks = [support.surfaces[i] for i in range(len(support.surfaces)) if i not in instance.sub_plane_ids]
                # in_contact_plane_masks = [support.surfaces[idx] for idx in instance.sub_plane_ids]

                # 2. Get vertical sub-planes of the supporting obj where the target obj is not placed on
                vertical_subplane_mask = get_subplanes_vertical(support,
                                                                direction,
                                                                instance)

                # 3. Get single obj freespace
                single_free_space, single_scores = get_single_obj_freespace(instance, support, vertical_subplane_mask)

                if j == 0:
                    inst_free_space = single_free_space
                    inst_scores = single_scores
                else:
                    inst_free_space = inst_free_space | single_free_space
                    inst_scores = (inst_scores + single_scores) - (inst_scores * single_scores)

            inst_scores /= inst_scores.max()

            if i == 0:
                free_space = inst_free_space
                scores = inst_scores
            else:
                free_space = free_space & inst_free_space
                scores *= inst_scores

    # 3. Assign scores to free space based on distance
    scores[~free_space] = 0
    scores /= scores.max()          # Divide by max value to set the max score to 1

    return free_space, scores


def compute_binary(support_and_tgts: Dict,
                   relations: str):
    """
    NOTE: Pipeline

    How to confirm that which objects are in betwen among target object candidates (instances)
    - Create a matching algorithm first
    - First, check if the supporting surface are the same
    - Then, check the distance of the objects
    - Confirm the pairs (groups) of objects that are the closest, which means that they are the real neighbors

    """
    assert relations in ["between"], ValueError("Only \"between\" relations are allowed.")

    # 1. Suppport object
    support = support_and_tgts["supporter"]
    objects_on = support_and_tgts.get("objects_on", None)

    # 2. Matching Algorithm
    valid_pairs = match_neighboring_pairs(objects_on)

    # 3. Define free space using pairs
    for i, instances in enumerate(valid_pairs):
        single_pair_freespace, single_pair_scores = get_pair_freespace(instances, support)

        if i == 0:
            free_space = single_pair_freespace
            scores = single_pair_scores
        else:
            free_space = free_space | single_pair_freespace
            scores = (scores + single_pair_scores) - (scores * single_pair_scores)

    # 3. Assign scores to free space based on distance
    scores[~free_space] = 0
    scores /= scores.max()          # Divide by max value to set the max score to 1

    return free_space, scores

# def compute_free_space(relation: str,
#                        obj_groups: List,
#                        all_freespace: np.ndarray):
#     for i, group in enumerate(obj_groups):
#         # 1. For perspective independent relations
#         if relation in ["besides", ]:
#             free_space_mask, free_space_scores = compute_next_to(group)

#         elif relation in ["below", "above"]:
#             free_space_mask, free_space_scores = compute_vertical(group, relation)

#         elif len(relation) == 0:            # on the
#             free_space_mask, free_space_scores = compute_vertical(group, "on")

#         elif relation == "between":
#             free_space_mask, free_space_scores = compute_binary(group, relation)

#         # 2. For perspective dependent relations
#         elif relation in ["left", ]:
#             pass

#         if i == 0:
#             all_mask = free_space_mask
#             all_scores = free_space_scores
#         else:
#             all_mask = np.logical_or(all_mask, free_space_mask)
#             all_scores += free_space_scores
#     all_scores /= all_scores.max()

#     return all_mask, all_scores


def compute_free_space(relation: str,
                       support: str,
                       targets: List[str],
                       objects_dict: Dict[str, List]):
    """
    
    """
    obj_groups = []

    # 1. Find appropriate support & supported-by object groups
    support_candidates = objects_dict[support]
    for support_inst in support_candidates:
        matched = {
            "supporter": support_inst,
            "objects_on": defaultdict(list)
        }

        for name, insts in support_inst.objects_on_surface.items():
            if name in targets:
                matched["objects_on"][name] = insts
        
        if len(matched["objects_on"]) != len(targets):
            continue

        obj_groups.append(matched)

    if len(obj_groups) == 0:
        return None, None

    # 2. Get free space
    for i, group in enumerate(obj_groups):
        print(relation)
        # 1. For perspective independent relations
        if relation in ['nearby', 'next to', 'adjacent to']:
            free_space_mask, free_space_scores = compute_next_to(group)

        elif relation in ["below", "above"]:
            free_space_mask, free_space_scores = compute_vertical(group, relation)

        elif len(relation) == 0:            # on the
            free_space_mask, free_space_scores = compute_vertical(group, "on")

        elif relation == "between":
            free_space_mask, free_space_scores = compute_binary(group, relation)

        # 2. For perspective dependent relations
        elif relation in ["left", ]:
            pass
        
        print(free_space_mask.sum(), free_space_scores.max(), free_space_scores.min())
        if i == 0:
            all_mask = free_space_mask
            all_scores = free_space_scores
        else:
            all_mask = np.logical_or(all_mask, free_space_mask)
            all_scores = (all_scores + free_space_scores) - (all_scores * free_space_scores)

    all_scores = all_scores / all_scores.max()

    return all_mask, all_scores