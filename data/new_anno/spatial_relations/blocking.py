import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPoint
from shapely import prepared
from shapely.strtree import STRtree
import alphashape
from scipy.spatial import cKDTree
import fpsample

from dataloader.objects import pcObject

def is_wall_point_blocking_segment(query_point, obj_point, wall_points, threshold=0.05):
    """
    Returns True if any wall point is within `threshold` of the segment and
    projects onto the segment (not just the extended line).
    """
    seg_vec = obj_point - query_point
    seg_len_sq = np.dot(seg_vec, seg_vec)
    
    for wall_pt in wall_points:
        vec_to_wall = wall_pt - query_point
        t = np.dot(vec_to_wall, seg_vec) / seg_len_sq  # projection factor

        if 0 <= t <= 1:  # only consider projections onto the segment
            proj_pt = query_point + t * seg_vec
            if np.linalg.norm(wall_pt - proj_pt) <= threshold:
                return True
    return False


def check_object_blocked_by_wall(query_point, object_points, wall_points, threshold=0.05):
    for obj_point in object_points:
        if is_wall_point_blocking_segment(query_point, obj_point, wall_points, threshold):
            return True
    return False


import numpy as np

def check_object_blocked_by_wall_vectorized(query_point, object_points, wall_points, threshold=0.05):
    """
    Vectorized version: Checks if any wall point blocks the segment between the query point
    and any of the object points.
    """
    # [M, 2] - from query to object
    seg_vecs = object_points - query_point  # shape (M, 2)
    seg_len_sq = np.sum(seg_vecs**2, axis=1)  # shape (M,)

    # [N, 2] - from query to wall points
    vec_to_walls = wall_points - query_point  # shape (N, 2)

    # Compute projection t for each wall point to each object segment: [N, M]
    t = np.einsum('nd,md->nm', vec_to_walls, seg_vecs) / seg_len_sq  # shape (N, M)

    # Keep only t values in [0, 1]
    valid_mask = (t >= 0) & (t <= 1)  # shape (N, M)

    if not np.any(valid_mask):
        return False

    # Project points on the segments: [N, M, 2]
    proj_pts = query_point + t[..., None] * seg_vecs[None, :, :]  # shape (N, M, 2)

    # Compute distances from wall points to projection points
    wall_pts_expanded = wall_points[:, None, :]  # shape (N, 1, 2)
    dists = np.linalg.norm(wall_pts_expanded - proj_pts, axis=2)  # shape (N, M)

    # Check if any distance is less than threshold for valid projections
    blocking = (dists <= threshold) & valid_mask
    return np.any(blocking)


def filter_outlier(points: np.ndarray,
                   object_points: np.ndarray,
                   wall_points: np.ndarray,
                   threshold: float = 0.01):
    """
    Filter outlier points in the free space
    e.g. free space behind the wall, too far points, ...

    Args:
        points (np.ndarray) : All points on the free space. Shape `(N,2)`
        object_points (np.ndarray) : All points of the main object in the description
        wall_points (np.ndarray) : All points of the wall
        threshold (float) : threshold value for checking if the wall is blocking

    Returns:
        outlier_filtered_mask (np.ndarray[bool]) : Boolean mask indicating the remaining points in the free space
    """
    fps_idx = fpsample.bucket_fps_kdline_sampling(points,
                                                  n_samples=32,
                                                  h=3,
                                                  start_idx=0)
    
    fps_points = points[fps_idx]
    tree = cKDTree(fps_points)
    distances, ids = tree.query(points, k=1)

    blocked_fps_points_ids = []
    for i, pt in enumerate(fps_points):
        if check_object_blocked_by_wall_vectorized(pt, object_points, wall_points, threshold):
            blocked_fps_points_ids.append(i)

    outlier_filtered_mask = np.isin(ids, blocked_fps_points_ids)
    return outlier_filtered_mask

    