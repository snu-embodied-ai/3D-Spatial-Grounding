import numpy as np
import random
from pxr import UsdGeom, UsdLux, Sdf, Gf
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.prims import GeometryPrim, RigidPrim
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import create_prim, get_prim_at_path


def generate_ground_plane(prim_path="/World/GroundPlane", is_visible=False):
    GroundPlane(prim_path=prim_path)
    ground_prim = get_prim_at_path(prim_path)
    if not is_visible:
        UsdGeom.Imageable(ground_prim).MakeInvisible()
    return ground_prim


def generate_light(prim_path="/World/DistantLight"):
    stage = get_current_stage()
    light = UsdLux.DistantLight.Define(stage, Sdf.Path(prim_path))
    light.CreateIntensityAttr(3000)
    light.CreateAngleAttr(30.0)
    light.CreateColorAttr((1.0, 1.0, 1.0))
    return light


def generate_table(
        prim_path="/World/Table",
        positions=np.array([[0.0, 0.0, 0.05]]),
        scales=np.array([[1.2, 0.8, 0.1]]),
    ):
    create_prim(prim_path=prim_path, prim_type="Cube", attributes={"size": 1.0})

    table = RigidPrim(prim_path)
    table.set_local_scales(scales=scales)
    table.set_world_poses(positions=positions)
    GeometryPrim(prim_path).apply_collision_apis()
    
    return table


TABLE_COLOR_DARK_GRAY = (0.1, 0.1, 0.1)
TABLE_COLOR_LIGHT_GRAY = (0.6, 0.6, 0.6)
TABLE_COLOR_WHITE = (1.0, 1.0, 1.0)
TABLE_COLOR_WOOD_BROWN = (0.5, 0.25, 0.1)

TABLE_COLORS = [
    TABLE_COLOR_DARK_GRAY,
    TABLE_COLOR_LIGHT_GRAY,
    TABLE_COLOR_WHITE,
    TABLE_COLOR_WOOD_BROWN,
]


class Table:
    def __init__(self, prim_path="/World/Table"):
        self.prim_path = prim_path
    
    def set_random_color(self):
        color = random.choice(TABLE_COLORS)
        self.set_color(color)

    def set_color(self, color):
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(self.prim_path)
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])