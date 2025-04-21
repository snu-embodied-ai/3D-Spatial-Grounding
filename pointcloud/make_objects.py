import carb
import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.prims import RigidPrim, GeometryPrim
from isaacsim.core.prims import XFormPrim

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

# === Banana ===
usd_path = assets_root_path + "/Isaac/Props/YCB/Axis_Aligned/011_banana.usd"
prim_path = "/World/Banana"
add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

banana = RigidPrim(prim_path)
banana.set_world_poses(positions=np.array([[0.15, -0.1, 0.15]]))
GeometryPrim(prim_path).apply_collision_apis()

# === Scissors ===
usd_path = assets_root_path + "/Isaac/Props/YCB/Axis_Aligned/037_scissors.usd"
prim_path = "/World/Scissors"
add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

scissors = RigidPrim(prim_path)
scissors.set_world_poses(positions=np.array([[-0.15, -0.1, 0.15]]))
GeometryPrim(prim_path).apply_collision_apis()

# === Mug ===
usd_path = assets_root_path + "/Isaac/Props/YCB/Axis_Aligned/025_mug.usd"
prim_path = "/World/Mug"
add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

mug = RigidPrim(prim_path)
mug.set_world_poses(positions=np.array([[0.15, 0.2, 0.15]]))
GeometryPrim(prim_path).apply_collision_apis()

# === Blue Block ===

usd_path = assets_root_path + "/Isaac/Props/Blocks/blue_block.usd"
prim_path = "/World/BlueBlock"
add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

blue_block = RigidPrim(prim_path)
blue_block.set_world_poses(positions=np.array([[-0.15, 0.2, 0.15]]))
GeometryPrim(prim_path).apply_collision_apis()
