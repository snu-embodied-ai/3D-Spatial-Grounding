import os
import numpy as np
import omni.replicator.core as rep
from pxr import UsdGeom, Usd, Sdf
from PIL import Image
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import XFormPrim


CAMERA_POSITIONS = [
    [ 1.7,  0.0, 1.7],
    [-1.8,  1.3, 1.3],
    [-1.8, -1.3, 1.3],
]

CAMERA_ORIENTATIONS = [
    [0.65328, 0.2706,   0.2706,   0.65328],
    [0.46194, 0.19134, -0.46194, -0.73254],
    [0.73254, 0.46194, -0.19134, -0.46194],
]


class Camera:
    def __init__(self, index, position, orientation,
                 focal_length, horizontal_aperture, vertical_aperture,
                 width, height):
        self.index = index
        self.position = position
        self.orientation = orientation
        self.focal_length = focal_length
        self.horizontal_aperture = horizontal_aperture
        self.vertical_aperture = vertical_aperture
        self.width = width
        self.height = height
        self.prim_path = f"/World/Camera_{index}"
        self.prim = None

    def setup(self):
        self._generate()
        self._set_intrinsic()
        self._apply_transform()
    
    def _generate(self):
        stage = get_current_stage()
        self.prim = UsdGeom.Camera.Define(stage, Sdf.Path(self.prim_path))
    
    def _set_intrinsic(self):
        self.prim.CreateFocalLengthAttr(self.focal_length)
        self.prim.CreateHorizontalApertureAttr(self.horizontal_aperture)
        self.prim.CreateVerticalApertureAttr(self.vertical_aperture)
    
    def _apply_transform(self):
        cam_xform = XFormPrim(self.prim_path)
        cam_xform.set_world_poses(
            positions=np.array([self.position], dtype=np.float32),
            orientations=np.array([self.orientation], dtype=np.float32)
        )

    def capture(self, output_dir, scene_index):
        render_product = self._create_render_product()

        depth_annotator = self._get_depth_annotator(render_product)
        rgb_annotator = self._get_rgb_annotator(render_product)

        rep.orchestrator.step()

        self._save_depth(depth_annotator, output_dir, scene_index)
        self._save_rgb(rgb_annotator, output_dir, scene_index)
    
    def _create_render_product(self):
        camera = rep.get.prim_at_path(self.prim_path)
        return rep.create.render_product(camera, (self.width, self.height))
    
    def _get_depth_annotator(self, render_product):
        depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        depth_annotator.attach([render_product])
        return depth_annotator

    def _get_rgb_annotator(self, render_product):
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])
        return rgb_annotator
    
    def _save_depth(self, annotator, output_dir, scene_index):
        data = annotator.get_data()
        if data is None:
            return
        path = os.path.join(output_dir, f"depth_frame_{scene_index}_{self.index}.npy")
        np.save(path, data)

    def _save_rgb(self, annotator, output_dir, scene_index):
        data = annotator.get_data()
        if data is None:
            return
        img = Image.fromarray(np.array(data, dtype=np.uint8))
        path = os.path.join(output_dir, f"rgb_frame_{scene_index}_{self.index}.png")
        img.save(path)

    def save_pose(self, output_dir):
        matrix = self._get_world_matrix()
        matrix = self._ensure_affine(matrix)
        self._save_pose(matrix, output_dir)
    
    def _get_world_matrix(self):
        prim = self.prim.GetPrim()
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        world_matrix = xform_cache.GetLocalToWorldTransform(prim)
        return np.array(world_matrix)

    def _ensure_affine(self, matrix):
        last_row = matrix[3]
        if abs(last_row[3] - 1.0) < 1e-8 and np.allclose(last_row[:3], 0, atol=1e-6):
            return matrix
        return matrix.T

    def _save_pose(self, matrix, output_dir):
        path = os.path.join(output_dir, f"pose_{self.index}.npy")
        np.save(path, matrix)


class CameraManager:
    def __init__(self, focal_length=50.0, horizontal_aperture=36, vertical_aperture=27,
                 width=1280, height=960):
        self.cameras = [
            Camera(i, position, orientation,
                   focal_length, horizontal_aperture, vertical_aperture,
                   width, height)
            for i, (position, orientation) in enumerate(zip(CAMERA_POSITIONS, CAMERA_ORIENTATIONS))
        ]

    def setup(self):
        for camera in self.cameras:
            camera.setup()

    def capture_all(self, out_dir, scene_count):
        for camera in self.cameras:
            camera.capture(out_dir, scene_count)

    def save_all_poses(self, out_dir):
        for camera in self.cameras:
            camera.save_pose(out_dir)
