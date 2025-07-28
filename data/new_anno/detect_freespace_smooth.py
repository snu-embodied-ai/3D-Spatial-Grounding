import numpy as np
from shapely.geometry import Polygon
import shapely
from matplotlib.path import Path as matPath

from dataloader.objects import pcObject
from spatial_relations.blocking import filter_outlier
from utils import sample_points, apply_gaussian_smoothing


def get_object_pairs(all_objects: list[pcObject],
                     walls: list[pcObject],
                     binary_threshold: float,
                     feasible_gap: float = 0.05,
                     pair_per_object: int = 2):

    hull_polygons = [Polygon(obj.points[:,:2]).convex_hull for obj in all_objects]

    wall_exists = len(walls) > 0

    if wall_exists:
        walls_xy = [wall.points[:,:2] for wall in walls]
        walls_xy = np.concatenate(walls_xy, axis=0)

    num_objects = len(all_objects)
    dist_mat = np.full((num_objects, num_objects), fill_value=1e6)

    for i, hull_i in enumerate(hull_polygons):
        obj_i_xy = all_objects[i].points[:,:2]
        for j in range(i+1, num_objects):  # Only compute for j > i to avoid redundant checks
            hull_j = hull_polygons[j]
            dist = hull_i.distance(hull_j)

            if feasible_gap <= dist <= binary_threshold:
                if wall_exists:
                    obj_j_xy = all_objects[j].points[:,:2]

                    blocked_mask = filter_outlier(obj_j_xy, obj_i_xy, walls_xy, threshold=0.01)

                    if not blocked_mask.any():
                        dist_mat[i, j] = dist
                        dist_mat[j, i] = dist  # Symmetric
                else:
                    dist_mat[i, j] = dist
                    dist_mat[j, i] = dist  # Symmetric

    # ================================================
    # TODO: TEST THIS CODE. CHECK VALIDITY!!!!!!!!
    dist_argsort = np.argsort(dist_mat, axis=1)
    pair_obj_ids = dist_argsort[:,:pair_per_object]
    pair_dists = np.take_along_axis(dist_mat, dist_argsort, axis=1)[:,:pair_per_object]

    # Make unique pairs where i < j
    pairs_with_dists = dict()
    for i in range(num_objects):
        for k in range(pair_per_object):
            j = pair_obj_ids[i, k]
            dist = pair_dists[i, k]

            if dist <= binary_threshold:
                pair = tuple(sorted((i, j)))
                # Only keep the smaller distance if pair already exists
                if pair not in pairs_with_dists or dist < pairs_with_dists[pair]:
                    pairs_with_dists[pair] = dist
    # ================================================

    # Convert dict to list of (i, j, distance)
    pair_list_with_dists = [(i, j, dist) for (i, j), dist in pairs_with_dists.items()]

    return pair_list_with_dists


def define_freespace(surface: pcObject) -> np.ndarray[bool]:
    """
    Finding the free space on the surface considering all objects

    Args:
        surface (pcObject) : Surface object supporting objects (e.g. floor, table)

    Returns:
        free_space (np.ndarray[bool]) : Boolean mask indicating the free space on the surface. Shape: `(num_surface_points,)`
    """
    surface_xy = surface.points[:,:2]
    free_space = np.ones(surface_xy.shape[0]).astype(bool)

    for obj in surface.children:
        # 0. Skip walls when defining the free space 
        if obj.final_label is None:
            continue
        else:
        
            # 1. Get convex hull of the object's projection as a Shapely polygon
            hull_polygon = Polygon(obj.points[:,:2]).convex_hull

            # 2. Find surface points outside the object's hull
            path = matPath(np.array(hull_polygon.exterior.coords))
            free_mask = ~path.contains_points(surface_xy)

            # free_mask = np.array([not hull_polygon.contains(Point(pt)) for pt in surface_xy])           # shape : (N,)
            
            # 3. Update free space
            free_space = np.logical_and(free_space, free_mask)

    return free_space.astype(bool)

def freespace_near_single(obj: pcObject,
                          surface: pcObject,
                          walls: list[pcObject],
                          free_space_mask: np.ndarray[bool],
                          threshold: float,
                          gaussian_mean_point: float = 0.2) -> np.ndarray[bool]:
    """
    Finding the free space on the surface close to the given object 

    Parameters
    ---
        obj : pcObject
            Object placed on the surface (non-walls)
        surface : pcObject
            Surface object supporting objects (e.g. floor, table)
        walls : list[pcObject]
            List of wall objects 
        free_space_mask : np.ndarray[bool]
            all free space on the surface object
        threshold : float
            distance threshold to exclude far-off regions
        gaussian_mean_point : float
            The mean of the Gaussian smoothing function, given as a float in (0, 1), representing the ratio relative to the threshold distance

    Returns
    ---
        final_mask : np.ndarray[bool]
            Boolean mask indicating the free space on the surface. Shape `(num_surface_points,)`
        free_space_scores : np.ndarray[float]
            Ground truth free space with smooth labels. The closer, more feasible points will have higher scores and farther and least feasbile points will have lower scores. Shape `(num_surface_points,)`
    """

    wall_exists = len(walls) > 0

    obj_xy = obj.points[:,:2]
    surface_xy = surface.points[:,:2]
    if wall_exists:
        walls_xy = [wall.points[:,:2] for wall in walls]
        walls_xy = np.concatenate(walls_xy, axis=0)

    # ==== 1. Get convex hull of the object's projection as a Shapely polygon ================
    hull_polygon = Polygon(obj_xy).convex_hull

    # ==== 2. COMPUTING DISTANCES ========================
    # Compute distance from each surface point to the hull boundary
    # dists = np.array([hull_polygon.exterior.distance(Point(pt)) for pt in surface_xy])          # shape : (N,)
    dists = shapely.distance(shapely.points(surface_xy), hull_polygon.exterior)

    # ==== 3. Select free points within the threshold distance from the hull ================
    within_threshold = dists <= threshold
    free_mask_within_threshold = np.logical_and(within_threshold,free_space_mask)

    if not wall_exists:
        final_mask = free_mask_within_threshold.astype(bool)
    else:
        # ==== 4. Select free points that aren't blocked by the walls ========
        # e.g. object -- | wall | -- free point
        free_points = surface_xy[free_mask_within_threshold]
        outlier_filtered_mask = filter_outlier(free_points, obj_xy, walls_xy)

        final_mask = np.zeros_like(free_mask_within_threshold, dtype=bool)
        final_mask[free_mask_within_threshold] = ~outlier_filtered_mask

        final_mask = final_mask.astype(bool)

    # ==== 5. Apply Gaussian smoothing to the labels =======
    mu = threshold * gaussian_mean_point
    free_space_scores = apply_gaussian_smoothing(mean=mu,
                                                 std=(threshold - mu) / 9.0,
                                                 # Distances for points in valid region
                                                 valid_dists=dists[final_mask],
                                                 final_mask=final_mask)

    return final_mask, free_space_scores



def freespace_near_pair(obj1: pcObject,
                        obj2: pcObject,
                        surface: pcObject,
                        pair_dist: float,
                        free_space_mask: np.ndarray[bool]):
    """
    Finding the free space on the surface close to the given object pair

    Parameters
    ---
        obj1 : pcObject
            Object 1 placed on the surface
        obj2 : pcObject
            Object 2 placed on the surface. (not blocked by walls)
        surface : pcObject
            Surface object supporting objects (e.g. floor, table)
        pair_dist : float
            The distance between the two objects
        free_space_mask : np.ndarray[bool]
            all free space on the surface object
    Returns
    ---
        free_mask_within_threshold : np.ndarray[bool]
            Boolean mask indicating the free space on the surface. Shape `(num_surface_points,)`
        free_space_scores : np.ndarray[float]
            Ground truth free space with smooth labels. The midpoints between two objects will have higher scores and farther and least feasbile points will have lower scores. Shape `(num_surface_points,)`
    """
    # 1. Get convex hull of the object's projection as a Shapely polygon
    obj1_xy = obj1.points[:,:2]
    obj2_xy = obj2.points[:,:2]
    hull_polygon_1 = Polygon(obj1_xy).convex_hull
    hull_polygon_2 = Polygon(obj2_xy).convex_hull

    surface_xy = surface.points[:,:2]

    # 2. Compute distance from each surface point to the hull boundary
    # dists_1 = np.array([hull_polygon_1.exterior.distance(Point(pt)) for pt in surface_xy])          # shape : (N,)
    dists_1 = shapely.distance(shapely.points(surface_xy), hull_polygon_1.exterior)
    # dists_2 = np.array([hull_polygon_2.exterior.distance(Point(pt)) for pt in surface_xy])          # shape : (N,)
    dists_2 = shapely.distance(shapely.points(surface_xy), hull_polygon_2.exterior)

    centroid = (obj1_xy.mean(axis=0) + obj2_xy.mean(axis=0)) / 2
    dist_from_centroid = np.linalg.norm(surface_xy - centroid, axis=1)

    # 3. Select free points within the threshold distance from the hull
    within_threshold = np.logical_and(dists_1 <= pair_dist, dists_2 <= pair_dist)
    free_mask_within_threshold = np.logical_and(within_threshold, free_space_mask)

    # 4. Apply Gaussian smoothing to the labels 
    # Centering the mean at midpoint between objects
    max_dist_from_centroid = dist_from_centroid.max()
    free_space_scores = apply_gaussian_smoothing(mean=0,
                                                 std=max_dist_from_centroid / 27.0,
                                                 valid_dists=dist_from_centroid[free_mask_within_threshold],
                                                 final_mask=free_mask_within_threshold)
    

    return free_mask_within_threshold, free_space_scores