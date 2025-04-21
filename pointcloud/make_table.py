from isaacsim.core.api.objects.ground_plane import GroundPlane
import numpy as np
from omni.isaac.core.prims import RigidPrim
from isaacsim.core.prims import GeometryPrim
import numpy as np
from omni.isaac.core.utils.prims import create_prim

GroundPlane(prim_path="/World/GroundPlane", z_position=0)
prim_path = "/World/Table"

# Cube (1m x 1m x 0.1m)
create_prim(
    prim_path=prim_path,
    prim_type="Cube",
    attributes={"size": 1.0},  # 1meter cube
)

table = RigidPrim(prim_path)
table.set_local_scale(np.array([1.0, 1.0, 0.1])) 
table.set_world_pose(position=np.array([[0, 0, 0.05]]))

RigidPrim("/World/Table")
prim = GeometryPrim("/World/Table")
prim.apply_collision_apis()





