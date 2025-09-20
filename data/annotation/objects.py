from collections import defaultdict
from typing import List
from typing_extensions import Self
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

class IsaacSimObject:
    def __init__(self,
                 obj_name: str,
                 instance_features: np.ndarray,
                 instance_mask: np.ndarray):
        
        self.name = obj_name
        # self.is_supporter = is_supporter
        self.inst_feats = instance_features
        self.inst_mask = instance_mask

        # 1. Generate pointcloud of the object
        self.pcd = self._create_pointcloud(self.inst_feats)

        # 2. Generate mesh of the object using the pointcloud
        self.mesh = self._create_mesh(self.pcd, obj_name == "table")

        # self._is_supporter = False
        self.surface_mask = None

        # # 3. If `is_supporter == True`, get surfaces where objects can be placed on
        # if is_supporter:
        #     self.surfaces = self.detect_support_surface()

        # 4. Properties for tree structure (supporter - objects on)
        self._objects_on_surface = defaultdict(list)
        self._supporter = None
        self._freespace = None
        self._valid_distances = None
        # self._in_contact_mask = None

    # @property
    # def is_supporter(self):
    #     return self._is_supporter
    
    # @is_supporter.setter
    # def is_supporter(self, is_supporter: bool):
    #     if is_supporter:
    #         self.surfaces = self.detect_support_surface()
    #     self._is_supporter = is_supporter


    @property
    def objects_on_surface(self):
        if len(self._objects_on_surface) == 0:
            return None
        else:
            return self._objects_on_surface
    
    def set_objects_on_surface(self, child_obj: Self):
        in_contact = is_in_contact(child_obj, self)
        if in_contact:
            self._objects_on_surface[child_obj.name].append(child_obj)
            child_obj._supporter = self
            # child_obj._sub_plane_ids = in_contact_mask
            return True
        else:
            print(f"Requested object {child_obj.name} is not in contact with current supporting object {self.name}")
            return False
        
    @property
    def supporting_object(self):
        assert self._supporter is not None, AttributeError("Should set supporting object first to access!!")
        return self._supporter
    
    @supporting_object.setter
    def supporting_object(self, supporter: Self):
        if is_in_contact(self, supporter):
            self._supporter = supporter
        else:
            raise ValueError(f"Requested object {supporter.name} is not in contact with current child object {self.name}")

    @property
    def freespace(self):
        assert self._freespace is not None, AttributeError("Must assign free space mask first to access")
        return self._freespace
    
    @freespace.setter
    def freespace(self, freespace_mask: np.ndarray):
        assert len(self.objects_on_surface) > 0, AttributeError("Cannot assign free space masks to non-supporter objects")
        self._freespace = freespace_mask

    @property
    def valid_distances(self):
        assert len(self.objects_on_surface) > 0, AttributeError("Cannot access valid distances on free space for non-supporter objects")
        assert self._valid_distances is not None, AttributeError("Must assign valid distances on free space first to access")
        return self._valid_distances
    
    @valid_distances.setter
    def valid_distances(self, valid_distances: np.ndarray):
        assert len(self.objects_on_surface) > 0, AttributeError("Cannot assign valid distances on free space to non-supporter objects")
        self._valid_distances = valid_distances

    # @property
    # def sub_plane_ids(self):
    #     assert self._sub_plane_id is not None, AttributeError(f"Sub planes in contact with object {self.name} is not assigned yet!")
    #     return self._sub_plane_ids


    def _create_mesh(self,
                     pcd: o3d.geometry.PointCloud,
                     is_table: bool = False) -> o3d.geometry.TriangleMesh:
        """
        Generate Open3D TriangleMesh object
        """
        # if is_table:
        #     mesh = pcd.compute_convex_hull()
        # else:
        #     radii = np.array([0.005, 0.01, 0.02, 0.04])
        #     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        #     mesh.compute_vertex_normals()
        mesh, _ = pcd.compute_convex_hull()
        return mesh

    def _create_pointcloud(self,
                           inst_feats: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Generate Open3D PointCloud object 
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inst_feats[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(inst_feats[:,3:])
        pcd.estimate_normals()
        
        return pcd
    
    def detect_support_surface(self,
                               tolerance: float = 0.3) -> np.ndarray:
        """
        Detecting the horizontal surfaces where objects can be placed on
        Don't use planar detection, 
        Just find by normal vector computation

        TODO:
        FIX all the related codes. Changed sub planes -> single surface mask
        """
        normalized = self.pcd.normals / np.linalg.norm(self.pcd.normals, axis=1, keepdims=True)
        z_orth = np.array([0,0,1]).reshape(-1,1)

        self.surface_mask = np.matmul(normalized, z_orth) > 1 - tolerance
        self.surface_mask = self.surface_mask.flatten()

        # # using all defaults
        # oboxes = self.pcd.detect_planar_patches(
        #     normal_variance_threshold_deg=60,
        #     coplanarity_deg=75,
        #     outlier_ratio=0.75,
        #     min_plane_edge_length=0,
        #     min_num_points=0,
        #     search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        # )
        
        # all_surface_masks = []
        # print(len(oboxes))
        # for obox in oboxes:
        #     normal = obox.R[:,2]
        #     print(normal)

        #     surface_normal = np.array([0,0,1])
        #     surface_check = np.matmul(normal, surface_normal)

        #     if surface_check > 1 - tolerance:
        #         included_ids = obox.get_point_indices_within_bounding_box(self.pcd.points)
        #         mask = np.zeros(self.inst_feats.shape[0])
        #         mask[included_ids] = True
        #         all_surface_masks.append(mask)


def is_in_contact(target_obj: IsaacSimObject,
                  support_obj: IsaacSimObject,
                  tolerance: float = 9e-3):
    """
    Check if the target object is in contact with the supporting object. In other words, if `True` the target object is located on the supporting object
    """
    # support_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(supporting_obj.pcd, voxel_size=voxel_size)
    # included_ids = support_voxel_grid.check_if_included(target_obj.pcd.points)

    # VOXELIZATION method -> maybe incorrect

    # voxelized_tgt = np.floor(target_obj.inst_feats[:,:3] / voxel_size)
    # voxelized_tgt = set(map(tuple, voxelized_tgt))

    # overlapped_surface_mask = []
    # for mask in surface_masks:
    #     pts = support_obj.inst_feats[:,:3][mask]
    #     vox = np.floor(pts / voxel_size)
        
    #     vox = set(map(tuple, vox))
    #     overlap = vox.intersection(voxelized_tgt)

    #     if len(overlap) > 0:
    #         overlapped_surface_mask.append(mask)
    
    # if len(overlapped_surface_mask) > 0:
    #     return overlapped_surface_mask
    # else:
    #     return False


    # Minimum distacne using cKDTree

    # 1. Use only the lower points of the target object
    tgt_points = target_obj.inst_feats[:,:3]
    tgt_height = tgt_points[:,2].max() - tgt_points[:,2].min()

    z_threshold = tgt_points[:,2].min() + tgt_height / 4
    query_tgt_points = tgt_points[tgt_points[:,2] < z_threshold]

    tree = cKDTree(query_tgt_points)

    # 2. Check regions on the supporting object that are in contact with the target object
    surface_points = support_obj.inst_feats[:,:3]

    # in_contact_plane_mask_ids = []      # Save the index of the mask in `support_obj.surfaces`, not the mask array to save memory
    # for i, mask in enumerate(support_obj.surfaces):
    #     plane_pts = surface_points[mask]
    #     dists, _ = tree.query(plane_pts, k=1)

    #     in_contact = np.min(dists) <= tolerance

    #     if in_contact:
    #         in_contact_plane_mask_ids.append(i)

    plane_pts = surface_points[support_obj.surface_mask]
    dists, _ = tree.query(plane_pts, k=1)

    print(np.min(dists))

    in_contact = np.min(dists) <= tolerance
    
    if in_contact.sum() > 0 :
        return True
    else:
        return False


"""
Ray Casting
1. Surface points
2. Distance query -> surface points to target objects
3. Filter regions where the distance is within threshold
4. PROBLEM : How to design the threshold..?
    a. First, define a arbitrary threshold for large surfaces
    b. Then, if the surface is smaller than 1.5 * threshold, cut off the threshold
    c. For binary relationship, the threshold must be the distance betweeen the target objects. No arbitrary thresholds pre-defined.
"""
