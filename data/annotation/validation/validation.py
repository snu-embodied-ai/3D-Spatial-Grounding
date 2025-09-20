import pickle
from pathlib import Path
import numpy as np
import open3d as o3d

DATA_DIR = Path("/media/jslee/219db482-9799-4615-841a-d8f46e93e50c/home/kykwon/SpatialUnderstanding/data/tabletop_dataset/stairs_second_opened_cabinet/global_clouds")

# 1. Load labelled result
with open("/media/jslee/219db482-9799-4615-841a-d8f46e93e50c/home/kykwon/SpatialUnderstanding/data/tabletop_dataset/annotation/output/freespace_label.pkl", 'rb') as f:
    label = pickle.load(f)

# 2. select index
SCENE_ID = 0
INDEX = 1

result = label[SCENE_ID][INDEX]
freespace = result["freespace_mask"]
scores = result["scores"]
desc = result["description"]
print(f"DESCRIPTION : {desc}")
print(scores.max(), scores.min(), scores.sum())

# 3. Create pointclouds for all data
pcd = o3d.io.read_point_cloud(str(DATA_DIR / f"pcd_{SCENE_ID}.ply"))
segments = np.load(str(DATA_DIR / f"segments_{SCENE_ID}.npy"))

pcd_semantics = o3d.geometry.PointCloud()
pcd_instances = o3d.geometry.PointCloud()
pcd_masked = o3d.geometry.PointCloud()
pcd_scores = o3d.geometry.PointCloud()

# Assign points
pcd_semantics.points = pcd.points
pcd_instances.points = pcd.points
pcd_masked.points = pcd.points
pcd_scores.points = pcd.points

# Assign colors
semantics_max = segments[:,0].max()
instance_max = segments[:,1].max()
semantics_rgb = np.repeat(segments[:,0:1] / semantics_max, repeats=3, axis=1)
instance_rgb = np.repeat(segments[:,1:] / instance_max, repeats=3, axis=1)
pcd_semantics.colors = o3d.utility.Vector3dVector(semantics_rgb)
pcd_instances.colors = o3d.utility.Vector3dVector(instance_rgb)

pcd_masked.colors = o3d.utility.Vector3dVector(np.repeat(freespace[:,None], repeats=3, axis=1))
pcd_scores.colors = o3d.utility.Vector3dVector(np.repeat(scores[:,None], repeats=3, axis=1))

# 4. Visualization
o3d.visualization.draw([
    {'name': 'original_pcd', 'geometry': pcd},
    {'name': 'semantic_segmentation', 'geometry': pcd_semantics},
    {'name': 'instance_segmentation', 'geometry': pcd_instances},
    {'name': 'free space', 'geometry': pcd_masked},
    {'name': 'free space scores', 'geometry': pcd_scores}
])