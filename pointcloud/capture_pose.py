import omni.replicator.core as rep
import numpy as np
import os
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom, Usd, Gf
from PIL import Image

OUTPUT_DIR = "./choi/pointcloud/data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def capture_depth_and_rgb(camera_num: int):
    camera_path = f"/World/Camera_{camera_num}"
    camera = rep.get.prim_at_path(camera_path)
    render_product = rep.create.render_product(camera, (640, 480))

    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    depth_annotator.attach([render_product])
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])

    rep.orchestrator.step_async()

    depth_data = depth_annotator.get_data()
    if depth_data is not None:
        np.save(os.path.join(OUTPUT_DIR, f"depth_frame_{camera_num}.npy"), depth_data)
    rgb_data = rgb_annotator.get_data()
    if rgb_data is not None:
        rgb_image = np.array(rgb_data, dtype=np.uint8)
        Image.fromarray(rgb_image).save(os.path.join(OUTPUT_DIR, f"rgb_frame_{camera_num}.png"))

def save_camera_pose(camera_num: int):
    """Camera pose save"""
    from pxr import UsdGeom
    stage = get_current_stage()
    camera_prim = stage.GetPrimAtPath(f"/World/Camera_{camera_num}")

    # GetLocalToWorldTransform()
    xformCache = UsdGeom.XformCache(Usd.TimeCode.Default())
    camera_world = xformCache.GetLocalToWorldTransform(camera_prim)

    np_mat = np.array(camera_world)
    # [Tx, Ty, Tz, 1]

    if abs(np_mat[3,3] - 1.0) < 1e-8 and np.allclose(np_mat[3,0:3], [0,0,0], atol=1e-6):
        pass
    else:
        np_mat = np_mat.T

    np.save(os.path.join(OUTPUT_DIR, f"pose_{camera_num}.npy"), np_mat)
    print(f"Camera {camera_num} pose saved:\n", np_mat)

if __name__ == "__main__":
    cam_id = 0
    capture_depth_and_rgb(cam_id)
    save_camera_pose(cam_id)
