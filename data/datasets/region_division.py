# Codes are taken from BPNet, CVPR'21 with some modifications
# https://github.com/wbhu/BPNet/blob/main/dataset/voxelizer.py

import collections
import numpy as np
from scipy.linalg import expm, norm

# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

class RegionDivider:
    def __init__(self,
                 region_size=0.05,
                 points_per_region=32,
                 region_threshold=8,
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

    def split_points(self, coords_scaled, coords_orig, feats):
        """
        Splittting input coordinates into groups. Creating a new dimension for the groups.
        ## Input
        - `coords_orig`: (N', 3), `feats` : (N', 3+@)
        ## Output
        - `divided_xyz`: (G, `points_per_region`, 3)
        - `divided_feats` : (G, `points_per_region`, 3+@)
        - `mask`: (G, `points_per_region`)
        - `centers`: (G, 3)
        """

        # all_features = np.concatenate((coords_orig, feats), axis=-1)    # (N', 6+@)

        unique_regions, region_idx, region_counts = np.unique(coords_scaled, axis=0, return_counts=True, return_inverse=True)

        cum_sum_counts = np.cumsum(region_counts)
        sort_feats = feats[np.argsort(region_idx)]

        lst_center = (unique_regions + 0.5) * self.region_size

        split = np.split(sort_feats, cum_sum_counts[:-1])
        all_regions, all_masks, all_centers = [], [], []
        # all_regions = np.zeros((len(region_counts), self.points_per_region, sort_feats.shape[-1]))
        # mask = np.zeros((len(region_counts), self.points_per_region))

        for i, region in enumerate(split):
            if len(region) >= self.points_per_region:
                sample_idx = np.random.choice(len(region), self.points_per_region, replace=False)
                all_regions.append(region[sample_idx])
                all_masks.append(np.ones(self.points_per_region))
            elif self.region_threshold < len(region) < self.points_per_region:
                points = np.zeros((self.points_per_region, sort_feats.shape[-1]))
                points[:len(region)] = region
                all_regions.append(points)
                
                mask = np.zeros(self.points_per_region)
                mask[:len(region)] = 1
                all_masks.append(mask)
            
            all_centers.append(lst_center[i])
            
        all_regions = np.stack(all_regions, axis=0)
        all_masks = np.stack(all_masks, axis=0)
        all_centers = np.stack(all_centers, axis=0)

        return all_regions[:, :, :3], all_regions, all_masks, all_centers


    def divide_regions(self, xyz:np.ndarray, feats:np.ndarray):
        """
        Dividing the whole scene into CUBIC regions

        ## Arguments
        - `xyz`: xyz coordinates of the input point cloud, of shape (N, 3)
        - `feats`: xyz + rgb + @ features of the input point cloud, of shape (N, 6+@). If normal vectors are included, they must be located at index 6, 7, 8 (right after the rgb features)
        - `return_idx`: Whether to return the indices of the remaining points, after clipping
        """
        # TODO : DO I NEED TO MAINTAIN `labels`? Seems like I don't need it.
        # Check if the input is valid
        assert xyz.shape[1] == 3 and xyz.shape[0] == feats.shape[0] and xyz.shape[0]
        
        if len(self.clip_bound) != 0:
            trans_aug_ratio = np.zeros(3)
            if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
                for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
                    trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

            clip_inds = self.clip(xyz, center=None, trans_aug_ratio=trans_aug_ratio)
            if clip_inds.sum():
                xyz, feats = xyz[clip_inds], feats[clip_inds]
                # if labels is not None:
                #     labels = labels[clip_inds]

        # Get rotation and scale
        M_d, M_r = self.get_transformation_matrix()
        # Apply transformations
        rigid_transformation = M_d
        if self.use_augmentation:
            rigid_transformation = M_r @ rigid_transformation

        homo_coords = np.hstack((xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)))
        xyz_aug = np.floor(homo_coords @ rigid_transformation.T[:, :3])
        if self.use_augmentation and self.rotation_augmentation_bound is not None:
            rot_only_xyz_aug = homo_coords @ M_r.T[:, :3]
        else:
            rot_only_xyz_aug = xyz

        # Normal rotation
        if feats.shape[1] > 6:
            feats[:, 6:9] = feats[:, 6:9] @ (M_r[:3, :3].T)

        # Split points into regions by introducing a new dimension
        divided_xyz, divided_feats, mask, centers = self.split_points(xyz_aug, rot_only_xyz_aug, feats)
        
        return divided_xyz, divided_feats, mask, centers, M_r


# class Region(nn.Module):
#     def __init__(self, region_size: float, num_points: int):
#         super().__init__()
#         self.region_size = region_size
#         self.num_points = num_points

#     def forward(self, xyz: torch.Tensor, color: torch.Tensor):
#         """
#         ## Parameters
#         - `xyz`: (B, N, 3) - xyz coordinates of the input point cloud
#         - `color`: (B, N, 3) - rgb value of the input point cloud

#         ## Output 
#         - `center`: (B, G, 3) - coordinates of the center of each region.
#         - `features`: (B, G, 6) - features of the points 
#         """
#         tolerance = 1e-2                        # 1cm tolerance
#         batch_size, num_points, _ = xyz.size()
        
#         x_max, y_max, z_max = xyz.max(dim=1) + tolerance    # (B, 3)
#         x_min, y_min, z_min = xyz.min(dim=1) - tolerance    # (B, 3)

#         x = torch.arange(x_min, x_max, self.region_size)
#         y = torch.arange(y_min, y_max, self.region_size)
#         z = torch.arange(z_min, z_max, self.region_size)

#         # Save center coordinates of all regions
#         min_bound = torch.cartesian_prod(x,y,z)
#         center = min_bound + (self.region_size / 2)
#         max_bound = min_bound + (self.region_size / 2)

#         region_bounds = torch.cat((min_bound, max_bound), dim=-1)

#         x_num_cells = torch.ceil((x_max - x_min) / self.region_size)
#         y_num_cells = torch.ceil((y_max - y_min) / self.region_size)
#         z_num_cells = torch.ceil((z_max - z_min) / self.region_size)

#         # Save center coordinates of all regions
