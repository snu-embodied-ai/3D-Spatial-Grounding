# Codes are taken from BPNet, CVPR'21 with some modifications
# https://github.com/wbhu/BPNet/blob/main/dataset/voxelizer.py

import collections
import numpy as np
from scipy.linalg import expm, norm
import torch
import torch_fpsample

# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

def normalize_points(features: torch.Tensor):
    """
    `features` : (G, `points_per_region`, 6+@)
    """
    points = features[:,:,:3]
    centroid = torch.mean(points, dim=1)                  # (G, 3)
    points = points - centroid.unsqueeze(dim=1)
    scale = torch.max(torch.sqrt((points**2).sum(dim=-1)), dim=-1).values  # (G,)
    features[:,:,:3] = points / scale[:,None,None]

    return features, centroid, scale

class RegionDivider:
    def __init__(self,
                 region_size=0.05,
                 points_per_region=32,
                 region_threshold=8,
                 ground_height=0.04,
                 clip_bound=None,
                 use_augmentation=False,
                 rotation_augmentation_bound=None,
                 translation_augmentation_ratio_bound=None,
                 ):
        r"""
        ## Arguments
        - `region_size`: length of the region(cube)'s edge
        - `points_per_region`: number of points for each region
        - `clip_bound`: boundary of the voxelizer. Points outside the bound will be deleted. Expects either None or an array like ((-100, 100), (-100, 100), (-100, 100))
        - `use_augmentation`: Indicating usage of augmentation
        - `rotation_augmentation_bound`: None or ((np.pi / 6, np.pi / 6), None, None) for 3 axis. Use random order of x, y, z to prevent bias.
        - `translation_augmentation_bound`: ((-5, 5), (0, 0), (-10, 10))
        """
        self.region_size = region_size
        self.points_per_region = points_per_region
        self.region_threshold = region_threshold
        self.ground_height = ground_height
        self.clip_bound = clip_bound

        # Properties for augmentation
        self.use_augmentation = use_augmentation
        
        if rotation_augmentation_bound is not None:
            self.rotation_augmentation_bound = np.zeros(6)
            for i in range(len(rotation_augmentation_bound)):
                self.rotation_augmentation_bound[i] = np.pi * eval(str(rotation_augmentation_bound[i]))
            self.rotation_augmentation_bound = self.rotation_augmentation_bound.reshape((3,2))
        else:
            self.rotation_augmentation_bound = None

        if translation_augmentation_ratio_bound is not None:
            self.translation_augmentation_ratio_bound = np.asarray(translation_augmentation_ratio_bound).reshape((3,2))
        else:
            self.translation_augmentation_ratio_bound = None

    def get_transformation_matrix(self):
        division_matrix, rotation_matrix = np.eye(4), np.eye(4)
        # Get clip boundary from config or pointcloud.
        # Get inner clip bound to crop from.

        # Transform pointcloud coordinate to voxel coordinate.
        # 1. Random rotation
        rot_mat = np.eye(3)
        if self.use_augmentation and self.rotation_augmentation_bound is not None:
            if isinstance(self.rotation_augmentation_bound, collections.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)
                    rot_mats.append(M(axis, theta))
                # Use random order
                np.random.shuffle(rot_mats)
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
            else:
                raise ValueError()
        rotation_matrix[:3, :3] = rot_mat


        # 2. Scale and translate to the voxel space.
        scale = 1 / self.region_size
        np.fill_diagonal(division_matrix[:3, :3], scale)

        # Get final transformation matrix.
        return division_matrix, rotation_matrix
    
    def clip(self, coords, center=None, trans_aug_ratio=None):
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        lim = self.clip_bound
        if trans_aug_ratio is not None:
            trans = np.multiply(trans_aug_ratio, bound_size)
            center += trans
        # Clip points outside the limit
        clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) &
                     (coords[:, 0] < (lim[0][1] + center[0])) &
                     (coords[:, 1] >= (lim[1][0] + center[1])) &
                     (coords[:, 1] < (lim[1][1] + center[1])) &
                     (coords[:, 2] >= (lim[2][0] + center[2])) &
                     (coords[:, 2] < (lim[2][1] + center[2])))
        return clip_inds

    def split_points(self, coords_scaled, coords_orig, feats, labels):
        """
        Splitting input coordinates into groups. Creating a new dimension for the groups.
        ## Input
        - `coords_orig`: (N', 3), `feats` : (N', 3+@), `labels` : (N`, 1)
        ## Output
        - `divided_xyz`: (G, `points_per_region`, 3)
        - `divided_feats` : (G, `points_per_region`, 3+@)
        - `mask`: (G, `points_per_region`)
        - `centers`: (G, 3)
        - `scales`: (G, )
        - `labels`: (G, `points_per_region`, 1)
        """

        unique_regions, region_idx, region_counts = torch.unique(coords_scaled, dim=0, return_counts=True, return_inverse=True)
        # centers = (unique_regions + 0.5) * self.region_size

        region_idx = torch.argsort(region_idx)
        sort_feats = feats[region_idx]
        sort_labels = labels[region_idx]

        split = torch.split(sort_feats, region_counts.tolist())
        label_split = torch.split(sort_labels, region_counts.tolist())

        all_regions, all_masks, all_labels = [], [], []

        for i, (region, label) in enumerate(zip(split, label_split)):
            if len(region) >= self.points_per_region:
                # To prevent under-sampling of the table surface
                is_table = region[:,2] <= self.ground_height
                if torch.any(is_table):
                    table_region = region[is_table]
                    table_labels = label[is_table]

                    if torch.any(~is_table):
                        object_region = region[~is_table]
                        object_labels = label[~is_table]

                        if object_region.size(0) <= self.points_per_region // 2:
                            table_region, fps_table_idx = torch_fpsample.sample(table_region.unsqueeze(dim=0), 
                                                                                self.points_per_region - object_region.size(0))
                            torch.rand()
                            table_labels = table_labels[fps_table_idx[0]]
                            table_region = table_region[0]
                        elif table_region.size(0) <= self.points_per_region // 2:
                            object_region, fps_object_idx = torch_fpsample.sample(object_region.unsqueeze(dim=0), 
                                                                                  self.points_per_region - table_region.size(0))
                            object_labels = object_labels[fps_object_idx[0]]
                            object_region = object_region[0]
                        else:
                            table_region, fps_table_idx = torch_fpsample.sample(table_region.unsqueeze(dim=0), self.points_per_region // 2)
                            table_labels = table_labels[fps_table_idx[0]]
                            table_region = table_region[0]

                            object_region, fps_object_idx = torch_fpsample.sample(object_region.unsqueeze(dim=0), 
                                                                                  self.points_per_region - table_region.size(0))
                            object_labels = object_labels[fps_object_idx[0]]
                            object_region = object_region[0]                                

                        fps_points = torch.cat((table_region, object_region), dim=0)
                        fps_labels = torch.cat((table_labels, object_labels), dim=0)
                    else:
                        # Region with only TABLE
                        table_region, fps_table_idx = torch_fpsample.sample(table_region.unsqueeze(dim=0), self.points_per_region)
                        fps_points = table_region[0]
                        fps_labels = table_labels[fps_table_idx[0]]

                else:
                    # Region with only OBJECTS
                    object_region, fps_object_idx = torch_fpsample.sample(region.unsqueeze(dim=0), self.points_per_region)
                    fps_labels = label[fps_object_idx[0]]
                    fps_points = object_region[0]
                
                all_regions.append(fps_points)
                all_masks.append(torch.ones(self.points_per_region))
                all_labels.append(fps_labels)

            elif self.region_threshold < len(region) < self.points_per_region:
                points = torch.zeros((self.points_per_region, region.shape[-1]))
                points[:len(region)] = region
                all_regions.append(points)
                
                mask = torch.zeros(self.points_per_region)
                mask[:len(region)] = 1
                all_masks.append(mask)

                empty_label = torch.zeros((self.points_per_region, label.shape[-1]))
                empty_label[:len(label)] = label
                all_labels.append(empty_label)
            else:
                pass
            
        all_regions = torch.stack(all_regions, dim=0)

        all_regions, all_centers, all_scale = normalize_points(all_regions)
        all_masks = torch.stack(all_masks, dim=0)
        # all_centers = centers[region_counts > self.region_threshold]
        all_labels = torch.stack(all_labels, dim=0)

        return all_regions[:, :, :3], all_regions, all_masks, all_centers, all_scale, all_labels


    def divide_regions(self, xyz: torch.Tensor, 
                       feats: torch.Tensor, 
                       label: torch.Tensor):
        """
        Dividing the whole scene into CUBIC regions

        ## Arguments
        - `xyz`: xyz coordinates of the input point cloud, of shape (N, 3)
        - `feats`: xyz + rgb + @ features of the input point cloud, of shape (N, 6+@). If normal vectors are included, they must be located at index 6, 7, 8 (right after the rgb features)
        - `label` : the label indicating the feasible points
        - `return_idx`: Whether to return the indices of the remaining points, after clipping
        """
        # Check if the input is valid
        assert xyz.shape[1] == 3 and xyz.shape[0] == feats.shape[0] == label.shape[0] and xyz.shape[0] and label.shape[1] == 1
        
        if len(self.clip_bound) != 0:
            trans_aug_ratio = torch.zeros(3)
            if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
                for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
                    trans_aug_ratio[axis_ind] = torch.rand(*trans_ratio_bound)

            clip_inds = self.clip(xyz, center=None, trans_aug_ratio=trans_aug_ratio)
            if clip_inds.sum():
                xyz, feats, label = xyz[clip_inds], feats[clip_inds], label[clip_inds]

        # Get rotation and scale
        M_d, M_r = self.get_transformation_matrix()
        # Apply transformations
        rigid_transformation = M_d
        if self.use_augmentation:
            rigid_transformation = M_r @ rigid_transformation

        homo_coords = torch.hstack([xyz, torch.ones((xyz.shape[0], 1))])
        xyz_aug = torch.floor(homo_coords @ rigid_transformation.T[:, :3])
        if self.use_augmentation and self.rotation_augmentation_bound is not None:
            rot_only_xyz_aug = homo_coords @ M_r.T[:, :3]
        else:
            rot_only_xyz_aug = xyz

        # Normal rotation
        if feats.shape[1] > 6:
            feats[:, 6:9] = feats[:, 6:9] @ (M_r[:3, :3].T)

        # Split points into regions by introducing a new dimension
        divided_xyz, divided_feats, mask, centers, scales, divided_label = self.split_points(xyz_aug, rot_only_xyz_aug, feats, label)
        
        return divided_xyz, divided_feats, divided_label, mask, centers, scales, M_r
