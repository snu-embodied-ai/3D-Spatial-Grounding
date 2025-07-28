import open3d as o3d
import numpy as np
from PIL import Image
import argparse
import os


POSE_DIR = "./results/knife_unique_aligned/pose"
RGBD_DIR = "./results/knife_unique_aligned/rgbd"
OUTPUT_DIR = "./results/knife_unique_aligned/global_clouds"


FOCAL_LENGTH = 50
HORIZONTAL_APERTURE = 36
VERTICAL_APERTURE = 27
IMAGE_W = 1280
IMAGE_H = 960

def rotate_around_x_axis(theta_deg):
    theta = np.deg2rad(theta_deg)
    cos = np.cos(theta)
    sin = np.sin(theta)

    return np.array([
        [1,   0,    0, 0],
        [0, cos, -sin, 0],
        [0, sin,  cos, 0],
        [0,   0,    0, 1]
    ])

def create_pointcloud_from_scene_and_camera(scene_index, camera_index):
    depth = load_depth(scene_index, camera_index)
    rgb = load_rgb(scene_index, camera_index)
    pose = load_camera_pose(camera_index)

    points = get_points_from_depth(depth)
    colors = get_colors_from_rgb(rgb)

    points, colors = filter_valid_points_and_colors(points, colors)

    pcd = create_pointcloud_from_points_and_colors(points, colors)
    pcd = transform_pointcloud_to_absolute_coordinate(pcd, pose)

    return pcd

def load_depth(scene_index, camera_index):
    return np.load(f"{RGBD_DIR}/depth_frame_{scene_index}_{camera_index}.npy")

def load_rgb(scene_index, camera_index):
    return np.array(Image.open(f"{RGBD_DIR}/rgb_frame_{scene_index}_{camera_index}.png"))[:, :, :3]

def load_camera_pose(camera_index):
    return np.load(f"{POSE_DIR}/pose_{camera_index}.npy")

def transform_intrinsic_to_pixel_scale():
    fx = (FOCAL_LENGTH / HORIZONTAL_APERTURE) * IMAGE_W
    fy = (FOCAL_LENGTH / VERTICAL_APERTURE) * IMAGE_H
    cx = IMAGE_W / 2
    cy = IMAGE_H / 2

    return fx, fy, cx, cy

def get_points_from_depth(depth):
    fx, fy, cx, cy = transform_intrinsic_to_pixel_scale()

    yy, xx = np.meshgrid(np.arange(IMAGE_H), np.arange(IMAGE_W), indexing='ij')

    x_norm = (xx - cx) / fx
    y_norm = (yy - cy) / fy
    dir_norm = np.sqrt(x_norm**2 + y_norm**2 + 1.0)

    z = depth / dir_norm
    x = x_norm * z
    y = y_norm * z
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    return points

def get_colors_from_rgb(rgb):
    return rgb.reshape(-1, 3).astype(np.float32) / 255.0

def filter_valid_points_and_colors(points, colors):
    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    points = points[valid_mask]
    colors = colors[valid_mask]
    return points, colors

def create_pointcloud_from_points_and_colors(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def transform_pointcloud_to_absolute_coordinate(pcd, pose):
    pcd.transform(rotate_around_x_axis(180))
    pcd.transform(pose)

    return pcd


def main(begin, end):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for scene in range(begin, end):

        cams = [0, 1, 2]
        merged_pcd = o3d.geometry.PointCloud()

        for c in cams:
            pcd = create_pointcloud_from_scene_and_camera(scene, c)
            merged_pcd += pcd

        o3d.io.write_point_cloud(f"{OUTPUT_DIR}/pcd_{scene}.ply", merged_pcd)
        print(f"[main()] Scene {scene} done")
    # for debug
    o3d.visualization.draw_geometries([merged_pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=1)
    parser.add_argument("--end", type=int, default=1000)
    args = parser.parse_args()
    main(args.begin, args.end)
