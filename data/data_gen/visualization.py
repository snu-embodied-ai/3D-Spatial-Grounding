import numpy as np
import open3d as o3d
import os
import argparse
from util import GraspNetObject, Surface, Description

def main(args):
    mesh = o3d.io.read_point_cloud(f"/media/jslee/219db482-9799-4615-841a-d8f46e93e50c/home/kykwon/SpatialUnderstanding/data/synthetic_tabletop/8objects/tabletop_{args.index}.ply/tabletop_{args.index}.ply")

    label = np.load(f"/media/jslee/219db482-9799-4615-841a-d8f46e93e50c/home/kykwon/SpatialUnderstanding/data/synthetic_tabletop/8objects/tabletop_{args.index}.ply/{args.index}_label_grid_size_{args.grid_size}.npy")

    occupancy = np.load(f"/media/jslee/219db482-9799-4615-841a-d8f46e93e50c/home/kykwon/SpatialUnderstanding/data/synthetic_tabletop/8objects/tabletop_1.ply/1_occupancy_grid_grid_size_0.002.npy")

    x_idx, y_idx = np.where(label)
    x_occ, y_occ = np.where(occupancy)

    grid_size = args.grid_size
    
    x = x_idx * grid_size
    y = y_idx * grid_size
    z = np.ones_like(x) * 0.04

    x_o = x_occ * grid_size
    y_o = y_occ * grid_size
    z_o = np.ones_like(x_o) * 0.04

    points = o3d.utility.Vector3dVector(np.stack((x,y,z), axis=-1))
    points_occ = o3d.utility.Vector3dVector(np.stack((x_o,y_o,z_o), axis=-1))

    label_pcd = o3d.geometry.PointCloud(points=points)
    label_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points) / 2)
    occ_pcd = o3d.geometry.PointCloud(points=points_occ)
    occ_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points) / 4)

    
    if args.env == "remote":
        ev = o3d.visualization.ExternalVisualizer(timeout=2000000)
        draw = ev.draw
        draw([{"name": "scene", "geometry": mesh}, {"name": "label", "geometry": label_pcd}, {"name": "occupancy_grid", "geometry": occ_pcd}])
    elif args.env == "local":
        o3d.visualization.draw(mesh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=1, type=int)
    parser.add_argument("--env", default="remote", type=str)
    parser.add_argument("--grid_size", default=0.002, type=float)

    args = parser.parse_args()

    main(args)