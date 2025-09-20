from typing import Dict, List, Optional

import open3d as o3d
import numpy as np
from objects import IsaacSimObject
from utils import apply_gaussian_smoothing, point_cloud_distance, compute_distances

def get_all_freespace(instances: List[IsaacSimObject],
                      threshold: float = 8e-3):
    """
    
    """
    all_masks = []

    for i, inst in enumerate(instances):
        if inst.objects_on_surface is None:
            continue

        # Initial free space mask
        free_space_mask = np.zeros_like(inst.inst_mask, dtype=bool)
        valid_distances = np.full_like(inst.inst_mask, fill_value=-1, dtype=np.float32)

        # Generate query points where the objects can be placed (planar region)
        query_points = inst.inst_feats[:,:3][inst.surface_mask]

        # Compute signed distance for all planar surface points
        objects_on = []
        for obj in inst.objects_on_surface.values():
            objects_on.extend(obj)
        signed_distance = compute_distances(objects_on, query_points)

        # Filter out points that has smaller distance than threshold
        valid_query_points = signed_distance > threshold
        
        free_space_mask_flat = free_space_mask[inst.inst_mask]
        free_space_mask_flat[inst.surface_mask] = valid_query_points
        free_space_mask[inst.inst_mask] = free_space_mask_flat
        
        # TODO: Whether to save the valid signed distances..?
        # Maybe this can be used for multi-object relations
        mask_indices = np.where(inst.inst_mask)[0][inst.surface_mask][valid_query_points]
        valid_distances[mask_indices] = signed_distance[valid_query_points]

        all_masks.append(free_space_mask)
        inst.freespace = free_space_mask
        inst.valid_distances = valid_distances
    
    all_masks = np.any(all_masks, axis=0)

    return all_masks


def get_single_obj_freespace(target_obj: IsaacSimObject,
                             support_obj: IsaacSimObject,
                             plane_mask: np.ndarray = None,
                             threshold: float = 0.2,
                             peak: float = 0.05,
                             slope: float =  9.0):
    """
    
    """

    # 1. Get the free space mask on the supporting obj
    freespace_mask = support_obj.freespace

    # 2. Get surface mask (flat region on the supporting obj)
    if plane_mask is None:
        plane_mask = support_obj.surface_mask

    in_contact_freespace = np.zeros((support_obj.inst_feats.shape[0]), dtype=bool)
    in_contact_freespace[plane_mask] = freespace_mask[support_obj.inst_mask][plane_mask]
    query_pts = support_obj.inst_feats[:,:3][in_contact_freespace]
    
    # 3. Compute distances from freespace points to the target obj
    signed_distance = compute_distances([target_obj], query_pts)

    # 4. Cut off the points farther than the threshold
    valid_pts = signed_distance <= threshold
    single_obj_freespace = np.zeros_like(freespace_mask)
    idx = np.where(support_obj.inst_mask)[0][in_contact_freespace]
    single_obj_freespace[idx] = valid_pts

    # 5. Assign scores
    std = (threshold - peak) / slope
    scores = apply_gaussian_smoothing(mean=peak,
                                      std=std,
                                      valid_dists=signed_distance[valid_pts],
                                      final_mask=single_obj_freespace)
    
    return single_obj_freespace, scores


def get_pair_freespace(target_objs: List[IsaacSimObject],
                       support_obj: IsaacSimObject,
                       plane_masks: Optional[List[np.ndarray]] = None,
                       slope: float =  9.0):
    """
    
    """
    inst1, inst2 = target_objs

    # 1. Get the free space mask on the supporting obj

    # # 2. Get sub-plane ids where the target objs are in contact with the supporting obj
    # subplane_ids = set(inst1.sub_plane_ids).intersection(set(inst2.sub_plane_ids))
    # in_contact_plane_masks = np.any([support_obj.surfaces[id] for id in subplane_ids], axis=0)

    # in_contact_freespace = freespace_mask[support_obj.inst_mask][in_contact_plane_masks]
    freespace_mask = support_obj.freespace[support_obj.inst_mask]
    query_pts = support_obj.inst_feats[:,:3][freespace_mask]

    # 3. Compute distances from freespace points to the target instance
    signed_distance_inst1 = compute_distances([inst1], query_pts)
    signed_distance_inst2 = compute_distances([inst2], query_pts)

    # 4. Strict threshold to exclude points that are not between instances (behind, ..)
    pcd_dist = point_cloud_distance(inst1, inst2)
    valid_dist1 = np.logical_and(signed_distance_inst1 > 0, signed_distance_inst1 <= pcd_dist)
    valid_dist2 = np.logical_and(signed_distance_inst2 > 0, signed_distance_inst2 <= pcd_dist)
    is_valid = np.logical_and(valid_dist1, valid_dist2)

    final_freespace = np.zeros_like(support_obj.freespace)
    final_freespace[support_obj.freespace] = is_valid
    binary_valid_dist = (signed_distance_inst1 + signed_distance_inst2)[is_valid] / 2
    
    mean = binary_valid_dist.min()
    std = mean / slope

    scores = apply_gaussian_smoothing(mean=mean,
                                      std=std,
                                      valid_dists=binary_valid_dist,
                                      final_mask=final_freespace)
    
    return final_freespace, scores