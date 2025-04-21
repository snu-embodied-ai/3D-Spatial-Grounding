import argparse
import numpy as np
import open3d as o3d
from PIL import Image

SAVE_DIR = "data"

# depends on camera setting in isaac sim
FOCAL_LENGTH = 50
HORIZONTAL_APERTURE = 36
VERTICAL_APERTURE = 24

# depends on capture.py
IMAGE_W = 640
IMAGE_H = 480

def make_point_cloud(camera_num: int):
    depth = np.load(f'{SAVE_DIR}/depth_frame_{camera_num}.npy')
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    rgb = np.array(Image.open(f'{SAVE_DIR}/rgb_frame_{camera_num}.png'))
    rgb = rgb[:, :, :3]  # Remove alpha

    fx = (FOCAL_LENGTH / HORIZONTAL_APERTURE) * IMAGE_W
    fy = (FOCAL_LENGTH / VERTICAL_APERTURE) * IMAGE_H
    cx = IMAGE_W / 2
    cy = IMAGE_H / 2

    # 픽셀 망 만들기
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # 방향 벡터의 정규화 계수 계산 (radial 거리 → z축 거리)
    x_norm = (xx - cx) / fx
    y_norm = (yy - cy) / fy
    dir_norm = np.sqrt(x_norm**2 + y_norm**2 + 1.0)

    # 깊이를 Z축 방향으로 정사영
    z = depth / dir_norm  # ✅ 진짜 z 값

    # x, y 좌표 계산
    x = x_norm * z
    y = y_norm * z

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    valid = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    points = points[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(f"{SAVE_DIR}/output_pointcloud_colored_{camera_num}.ply", pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture RGB and depth from specified camera.")
    parser.add_argument("--camera_num", type=int, required=True, help="Camera number to capture from")
    args = parser.parse_args()

    camera_num = args.camera_num
    print(f"Selected camera number: {camera_num}")
    make_point_cloud(camera_num=camera_num)