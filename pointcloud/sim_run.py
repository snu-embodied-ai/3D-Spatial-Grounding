from isaacsim import SimulationApp

# === Start Simulation App ===
simulation_app = SimulationApp({"headless": True})

import os
from isaacsim.core.api import World
from isaacsim.storage.native import get_assets_root_path

from sim_utils.asset import AssetCatalog, LOCAL_ASSET_ROOT_PATH
from sim_utils.sampler import PositionSampler
from sim_utils.spawner import SceneBuilder
from sim_utils.camera import CameraManager
from sim_utils.logger import SceneLogger
from sim_utils.scene import Table, generate_ground_plane, generate_light, generate_table

# === Save Directory ===
RGBD_DIR = "./pointcloud/rgbd"
POSE_DIR = "./pointcloud/pose"
LOG_DIR = "./pointcloud/log"

# === Basic Scene ===
ground_prim = generate_ground_plane()
light_prim = generate_light()
table_prim = generate_table()
table = Table()

# === Asset, Sampler, Builer, Camera Manager ===
catalog = AssetCatalog(get_assets_root_path(), LOCAL_ASSET_ROOT_PATH)
sampler = PositionSampler()
builder = SceneBuilder(catalog, sampler)
cam_manager = CameraManager()
logger = SceneLogger(LOG_DIR)

# === Setup Camera & Save Poses ===
cam_manager.setup()
os.makedirs(POSE_DIR, exist_ok=True)
cam_manager.save_all_poses(POSE_DIR)

# === Main Simulation Loop ===
os.makedirs(RGBD_DIR, exist_ok=True)
world = World()
world.reset()

spawned = []
scene_index = 0
spawn_interval = 2

print("[Main] Simulation start")
for frame in range(1000):
    if frame % spawn_interval == 0:
        builder.clear_objects()
        table.set_random_color()

        world.step(render=True)
        builder.spawn_random()
        object_info = builder.get_spawned_object_info()
        logger.log(scene_index, object_info)
        print(f"[Main] Spawned {len(object_info)} objects at scene {scene_index}")

        for _ in range(15):
            world.step(render=True)
        cam_manager.capture_all(RGBD_DIR, scene_index)
        scene_index += 1

    world.step(render=True)

print("[Main] Simulation terminated")
simulation_app.close()