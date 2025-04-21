import open3d as o3d
import numpy as np
from PIL import Image
import os

SAVE_DIR = "./data"
OUTPUT_DIR = "./global_clouds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOCAL_LENGTH = 50
HORIZONTAL_APERTURE = 36
VERTICAL_APERTURE = 24
IMAGE_W = 640
IMAGE_H = 480

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

def create_pointcloud(camera_num: int):
    depth = np.load(f"{SAVE_DIR}/depth_frame_{camera_num}.npy")
    rgb = np.array(Image.open(f"{SAVE_DIR}/rgb_frame_{camera_num}.png"))[:, :, :3]
    pose = np.load(f"{SAVE_DIR}/pose_{camera_num}.npy")

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

    o3d.io.write_point_cloud(f"{OUTPUT_DIR}/pcd_{camera_num}.ply", pcd)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    axis.transform(pose)

    return pcd, axis

def create_grid(size=4.0, divisions=20):
    lines = []
    points = []

    step = size / divisions
    half = size / 2.0

    for i in range(divisions + 1):
        x = -half + i * step
        points.append([x, -half, 0])
        points.append([x, half, 0])
        lines.append([2 * i, 2 * i + 1])

        y = -half + i * step
        points.append([-half, y, 0])
        points.append([half, y, 0])
        lines.append([2 * (divisions + 1) + 2 * i, 2 * (divisions + 1) + 2 * i + 1])

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    return grid

def main():
    cams = [0, 1, 2]
    all_pcds = []
    all_axes = []

    for c in cams:
        pcd, axis = create_pointcloud(c)
        all_pcds.append(pcd)
        all_axes.append(axis)

    grid = create_grid(size=4.0, divisions=20)
    o3d.visualization.draw_geometries(all_pcds + all_axes + [grid])

if __name__ == "__main__":
    main()
