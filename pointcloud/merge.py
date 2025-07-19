import open3d as o3d
import numpy as np
from PIL import Image
import argparse
import os

NUM_SCENES = 1000
POSE_DIR = "./pose"
RGBD_DIR = "./rgbd"
OUTPUT_DIR = "./global_clouds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOCAL_LENGTH = 50
HORIZONTAL_APERTURE = 36
VERTICAL_APERTURE = 27
IMAGE_W = 1280
IMAGE_H = 960

def rotate_around_x(theta_deg: float):
    theta = np.deg2rad(theta_deg)
    cos = np.cos(theta)
    sin = np.sin(theta)

    return np.array([
        [1,   0,    0, 0],
        [0, cos, -sin, 0],
        [0, sin,  cos, 0],
        [0,   0,    0, 1]
    ])

def create_pointcloud(scene: int, camera_num: int):
    depth = np.load(f"{RGBD_DIR}/depth_frame_{scene}_{camera_num}.npy")
    rgb = np.array(Image.open(f"{RGBD_DIR}/rgb_frame_{scene}_{camera_num}.png"))[:, :, :3]
    pose = np.load(f"{POSE_DIR}/pose_{camera_num}.npy")

    fx = (FOCAL_LENGTH / HORIZONTAL_APERTURE) * IMAGE_W
    fy = (FOCAL_LENGTH / VERTICAL_APERTURE) * IMAGE_H
    cx = IMAGE_W / 2
    cy = IMAGE_H / 2

    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    x_norm = (xx - cx) / fx
    y_norm = (yy - cy) / fy
    dir_norm = np.sqrt(x_norm**2 + y_norm**2 + 1.0)

    z = depth / dir_norm
    x = x_norm * z
    y = y_norm * z

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    points = points[valid_mask]
    colors = colors[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.transform(rotate_around_x(180))
    pcd.transform(pose)

    return pcd

def main(begin, end):
    for scene in range(begin, end):

        cams = [0, 1, 2]
        merged_pcd = o3d.geometry.PointCloud()

        for c in cams:
            pcd = create_pointcloud(scene, c)
            merged_pcd += pcd

        o3d.io.write_point_cloud(f"{OUTPUT_DIR}/pcd_{scene}.ply", pcd)
        print(f"[main()] Scene {scene} done")
    # for debug
    o3d.visualization.draw_geometries([merged_pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=1)
    parser.add_argument("--end", type=int, default=1000)
    args = parser.parse_args()
    main(args.begin, args.end)
