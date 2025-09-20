from typing import List, Dict, Tuple
import itertools

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from objects import IsaacSimObject

# def point_cloud_distance(pc1, pc2):
#     tree1 = cKDTree(pc1)
#     dist, _ = tree1.query(pc2)
#     return dist.min()

def compute_distances(mesh_objects: List[IsaacSimObject],
                      query_pts: np.ndarray):
    """
    Returning signed distance
    """
    scene = o3d.t.geometry.RaycastingScene()

    mesh_ids = dict()
    for obj in mesh_objects:
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(obj.mesh)
        mesh_ids[scene.add_triangles(t_mesh)] = obj.name
    
    return scene.compute_signed_distance(query_pts.astype(np.float32)).numpy()

def point_cloud_distance(inst1: IsaacSimObject,
                         inst2: IsaacSimObject):
    signed_distances = compute_distances([inst1], inst2.inst_feats[:,:3])
    return np.absolute(signed_distances).min()

def apply_gaussian_smoothing(mean: float,
                             std: float,
                             valid_dists: np.ndarray[float],
                             final_mask: np.ndarray[bool]):
    """
    Apply gaussian smoothing to the labels
    Sum Normalization applied for matching the scale and changing to a probabilistc distribution
    """
    final_output = np.zeros_like(final_mask, dtype=float)

    scores = np.exp(-((valid_dists - mean) ** 2) / (2 * std ** 2))
    
    # Set the max score to 1
    scores /= scores.max()

    final_output[final_mask] = scores

    return final_output


def get_subplanes_vertical(support: IsaacSimObject,
                           relation: str,
                           target_instance: IsaacSimObject,
                           threshold: float = 0.003):
    """
    Get non-contact subplanes that are below/above the in-contact subplanes

    Parameters
    ---

    """
    assert relation in ["below", "above"], ValueError(f"{relation} is not supported for this function get_subplane_vertical() !!")

    # 1. Get the minimum z value of the the target instance to get in-contact surfaces
    z_min = target_instance.inst_feats[:,2].min()
    surface_pts = support.inst_feats[support.surface_mask]

    in_contact_regions = np.logical_and(surface_pts[:,2] <= z_min + threshold, surface_pts[:,2] >= z_min - threshold)
    in_contact_pts = surface_pts[in_contact_regions]

    if relation == "below":
        z_in_contact = in_contact_pts[:,2].min()
    elif relation == "above":
        z_in_contact = in_contact_pts[:,2].max()

    # 2. Get the non-contact subplanes that are below/above the in-contact subplanes
    non_contact_pts = surface_pts[~in_contact_regions]
    vertical_subplane_mask = np.zeros(support.inst_feats.shape[0], dtype=bool)
    mask = np.where(support.surface_mask)[0][~in_contact_regions]
    if relation == "below":
        vertical_subplane_mask[mask] = non_contact_pts[:,2] < z_in_contact
    elif relation == "above" :
        vertical_subplane_mask[mask] = non_contact_pts[:,2] > z_in_contact
    # vertical_subplane_masks = []
    # for mask in non_contact_subplanes:
    #     points = support.inst_feats[:,:3][mask]

    #     if relation == "below" and points.max() < z_in_contact:
    #         vertical_subplane_masks.append(mask)
    #     elif relation == "above" and points.min() > z_in_contact:
    #         vertical_subplane_masks.append(mask)

    return vertical_subplane_mask


def match_neighboring_pairs(objects: Dict[str, List[IsaacSimObject]],
                            tolerance: float = 8e-3) -> List[Tuple[IsaacSimObject, IsaacSimObject]]:
    """
    Match the neighboring instances in the given object dictionary
    If the keys of the dictionary are "apple" and "box", this function returns a list of "apple" instance - "box" instance pairs which are actually neighboring.

    """
    object_names = list(objects.keys())
    assert len(object_names) == 2, ValueError("Only pairwise comparison is considered for match_neighboring_pairs. Should pass only two objects")
    
    valid_pairs = []
    for inst1 in objects[object_names[0]]:
        for inst2 in objects[object_names[1]]:
            # 1. Check if two instances are on the same suppporting object
            if (inst1.supporting_object is not inst2.supporting_object):
                continue

            # 1-2. Check if the two instances are on the same plane (similar minimum z coordinate)
            z1_min = inst1.inst_feats[:,2].min()
            z2_min = inst2.inst_feats[:,2].min()
            if abs(z1_min - z2_min) > tolerance:
                continue

            # 2. Compute point cloud distance between two instances
            # pcd_dist = point_cloud_distance(inst1.inst_feats[:,:3], inst2.inst_feats[:,:3])
            pcd_dist = point_cloud_distance(inst1, inst2)

            objects_without_inst1 = objects[object_names[0]] + objects[object_names[1]]
            objects_without_inst1.remove(inst1)
            objects_without_inst2 = objects[object_names[0]] + objects[object_names[1]]
            objects_without_inst2.remove(inst2)

            inst1_to_mesh = compute_distances(objects_without_inst1, inst1.inst_feats[:,:3])
            inst1_to_mesh = np.absolute(inst1_to_mesh).min()
            inst2_to_mesh = compute_distances(objects_without_inst2, inst2.inst_feats[:,:3])
            inst2_to_mesh = np.absolute(inst2_to_mesh).min()

            diff1 = pcd_dist - inst1_to_mesh
            diff2 = pcd_dist - inst2_to_mesh
            if diff1 < tolerance and diff2 < tolerance:
                valid_pairs.append([inst1, inst2])

    if len(valid_pairs) > 0:
        return valid_pairs
    else:
        return False