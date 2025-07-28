import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree
import fpsample

@njit(parallel=True)
def check_object_blocked_by_wall_vectorized(query_points, object_points, wall_points, threshold=0.05):
    # """
    # Vectorized version: Checks if any wall point blocks the segment between the query point
    # and any of the object points.
    # """
    # # [M, 2] - from query to object
    # seg_vecs = object_points - query_point  # shape (M, 2)
    # seg_len_sq = np.sum(seg_vecs**2, axis=1)  # shape (M,)

    # # [N, 2] - from query to wall points
    # vec_to_walls = wall_points - query_point  # shape (N, 2)

    # # Compute projection t for each wall point to each object segment: [N, M]
    # t = np.einsum('nd,md->nm', vec_to_walls, seg_vecs) / seg_len_sq  # shape (N, M)

    # # Keep only t values in [0, 1]
    # valid_mask = (t >= 0) & (t <= 1)  # shape (N, M)

    # if not np.any(valid_mask):
    #     return False

    # # Project points on the segments: [N, M, 2]
    # proj_pts = query_point + t[..., None] * seg_vecs[None, :, :]  # shape (N, M, 2)

    # # Compute distances from wall points to projection points
    # wall_pts_expanded = wall_points[:, None, :]  # shape (N, 1, 2)
    # dists = np.linalg.norm(wall_pts_expanded - proj_pts, axis=2)  # shape (N, M)

    # # Check if any distance is less than threshold for valid projections
    # blocking = (dists <= threshold) & valid_mask
    # return np.any(blocking)
    """
    Numba-accelerated batch version.
    
    Args:
        query_points: (B, 2)
        object_points: (M, 2)
        wall_points: (N, 2)
        threshold: float
        
    Returns:
        blocked: (B,) bool array
    """
    B = query_points.shape[0]
    M = object_points.shape[0]
    N = wall_points.shape[0]

    blocked = np.zeros(B, dtype=np.bool_)

    for b in prange(B):
        qx, qy = query_points[b]

        for m in range(M):
            ox, oy = object_points[m]
            dx = ox - qx
            dy = oy - qy
            seg_len_sq = dx * dx + dy * dy + 1e-8

            for n in range(N):
                wx, wy = wall_points[n]
                vx = wx - qx
                vy = wy - qy

                t = (vx * dx + vy * dy) / seg_len_sq

                if 0 <= t <= 1:
                    px = qx + t * dx
                    py = qy + t * dy

                    dist_sq = (wx - px) ** 2 + (wy - py) ** 2
                    if dist_sq <= threshold ** 2:
                        blocked[b] = True
                        break  # wall point blocks the segment

            if blocked[b]:
                break  # no need to check other object points

    return blocked


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
    # fps_idx = fpsample.bucket_fps_kdline_sampling(points,
    #                                               n_samples=32,
    #                                               h=3,
    #                                               start_idx=0)
    
    # fps_points = points[fps_idx]
    # tree = cKDTree(fps_points)
    # distances, ids = tree.query(points, k=1)

    # blocked_fps_points_ids = []
    # for i, pt in enumerate(fps_points):
    #     if check_object_blocked_by_wall_vectorized(pt, object_points, wall_points, threshold):
    #         blocked_fps_points_ids.append(i)

    # outlier_filtered_mask = np.isin(ids, blocked_fps_points_ids)
    # return outlier_filtered_mask
    # FPS sampling
    if points.shape[0] < 32:
        fps_points = points
    else:
        fps_idx = fpsample.bucket_fps_kdline_sampling(points, n_samples=32, h=3, start_idx=0)
        fps_points = points[fps_idx]

    # Nearest neighbor search
    tree = cKDTree(fps_points)
    distances, ids = tree.query(points, k=1)

    # Blocking check with numba
    blocked_fps_mask = check_object_blocked_by_wall_vectorized(fps_points, object_points, wall_points, threshold)

    # Convert to indices
    blocked_fps_points_ids = np.where(blocked_fps_mask)[0]

    # Outlier mask
    outlier_filtered_mask = np.isin(ids, blocked_fps_points_ids)
    return outlier_filtered_mask
    