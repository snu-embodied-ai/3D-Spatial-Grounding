import numpy as np
import random
from .spawner import Spawner
from sim_utils.assets.asset import AssetCatalog
from sim_utils.samplers import PositionRandomSampler
from pxr import Sdf
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import XFormPrim


STANDUP_ORIENTATION = [np.cos(np.deg2rad(45)), -np.sin(np.deg2rad(45)), 0, 0]
TARGET_OBJECT_NAME = "banana"


class AlignedSpawner(Spawner):
    def __init__(self, catalog: AssetCatalog, sampler: PositionRandomSampler, num_of_objects):
        super().__init__(catalog, sampler, num_of_objects)
        self.target_object_name = TARGET_OBJECT_NAME
    
    def spawn_random(self):
        names = self.catalog.random_names(self.num_of_objects)
        positions = self.sampler.sample(self.num_of_objects)
        self._transform_to_aligned_names(names, positions)
        self.spawn(names, positions)
    
    def _transform_to_aligned_names(self, names, positions, threshold=0.07):
        aligned_target_line = positions[0][1]
        num_targets = 0
        for position in positions:
            if abs(position[1] - aligned_target_line) < threshold:
                num_targets += 1
            else:
                break

        names[:num_targets] = [self.target_object_name] * num_targets

    def spawn(self, names, positions):
        self.clear_objects()
        for i, (name, position) in enumerate(zip(names, positions)):
            prim_path = self._spawn_one(i, name, position)
            self.spawned_object_paths.append(prim_path)

    def _spawn_one(self, object_index, name, position):
        asset_path = self.catalog.get_path(name)
        parent_path = f"/World/Object_{object_index}"
        add_reference_to_stage(usd_path=asset_path, prim_path=parent_path)

        position = self._get_object_position(name, position)
        orientation = self._get_orientation(name)
        self._apply_transform(parent_path, position, orientation)
        self._append_info(name, position)
        return parent_path

    def _get_object_position(self, name, position):
        z = self.catalog.get_spawn_height(name)
        position = [position[0], position[1], z]
        return position

    def _get_orientation(self, name):
        if name.endswith("_u"):
            orientation = STANDUP_ORIENTATION
        else:
            orientation = [1, 0, 0, 0]
        return orientation
    
    def _apply_transform(self, parent_path, position, orientation):
        orientation = self._apply_random_z_rot(orientation)
        xform = XFormPrim(parent_path)
        xform.set_world_poses(
            positions=np.array([position], dtype=np.float32),
            orientations=np.array([orientation], dtype=np.float32)
        )
    
    def _apply_random_z_rot(self, orientation):
        theta = random.randint(0, 180)
        z_rot = [np.cos(np.deg2rad(theta/2)), 0, 0, np.sin(np.deg2rad(theta/2))]
        return self._quat_mul(z_rot, orientation)

    def _quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return [w, x, y, z]

    def _append_info(self, name, position):
        self.spawned_object_info.append({
            "name": name,
            "position": position,
        })
    
    def clear_objects(self):
        for prim_path in self.spawned_object_paths:
            delete_prim(Sdf.Path(prim_path))
        self.spawned_object_paths.clear()
        self.spawned_object_info.clear()

    def get_spawned_object_info(self):
        return self.spawned_object_info
