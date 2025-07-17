import numpy as np
import open3d as o3d
import os
import argparse
import glob
from tqdm import tqdm

def label_points(pcd: o3d.geometry.PointCloud,
                 label: np.array,
                 grid_size: float,
                 idx: str):
    points = np.asarray(pcd.points)
    points_to_grid = (points / grid_size).astype('int')

    # Ensure points are within bounds of label array
    max_x, max_y = label.shape
    x = points_to_grid[:, 0]
    y = points_to_grid[:, 1]
    z = points[:, 2]
    mask = (x >= 0) * (x < max_x) * (y >= 0) * (y < max_y) * (z > 0.039)

    print(f"START LABELING {idx}")

    valid_label = np.zeros(len(points_to_grid), dtype=np.uint8)
    valid_label[mask] = label[x[mask], y[mask]] > 0

    is_valid_to_rgb = np.pad(valid_label[:,None], ((0,0), (0,2)), constant_values=0)

    ply_label = o3d.geometry.PointCloud()
    ply_label.points = o3d.utility.Vector3dVector(points)
    ply_label.colors = o3d.utility.Vector3dVector(is_valid_to_rgb)

    return ply_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/home/kykwon/3D-Spatial-Grounding/data/synthetic_tabletop", type=str)
    parser.add_argument("--grid_size", default=0.002, type=float)

    args = parser.parse_args()

    for idx in tqdm(os.listdir(args.dataset_path)):
        if idx.endswith(".csv"):
            continue
        else:
            sample_dir = os.path.join(args.dataset_path, idx)
            sample_ply = o3d.io.read_point_cloud(glob.glob(os.path.join(sample_dir, "tabletop_*.ply"))[0])
            label = np.load(glob.glob(os.path.join(sample_dir, "*label*.npy"))[0], allow_pickle=True)

            ply_label = label_points(sample_ply, label, args.grid_size, idx)

            print(f"Saving Point Cloud Label {idx} ...")
            o3d.io.write_point_cloud(os.path.join(sample_dir, f"{idx}_segmentation.ply"), ply_label, write_ascii=True, print_progress=True)
    
    print("DONE LABELING POINT CLOUDS!!")