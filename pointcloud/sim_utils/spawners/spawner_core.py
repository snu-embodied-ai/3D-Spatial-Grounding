import numpy as np
import random
from isaacsim.core.prims import XFormPrim


STANDUP_ORIENTATION = [np.cos(np.deg2rad(45)), -np.sin(np.deg2rad(45)), 0, 0]


def apply_transform(parent_path, position, orientation):
    orientation = apply_random_z_rot(orientation)
    xform = XFormPrim(parent_path)
    xform.set_world_poses(
        positions=np.array([position], dtype=np.float32),
        orientations=np.array([orientation], dtype=np.float32)
    )
    
def apply_random_z_rot(orientation):
    theta = random.randint(0, 180)
    z_rot = [np.cos(np.deg2rad(theta/2)), 0, 0, np.sin(np.deg2rad(theta/2))]
    return quat_mul(z_rot, orientation)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return [w, x, y, z]
