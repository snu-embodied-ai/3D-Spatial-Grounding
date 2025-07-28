from abc import ABC, abstractmethod
from .spawner_core import apply_transform, STANDUP_ORIENTATION
from pxr import Sdf
from isaacsim.core.utils.prims import delete_prim
from isaacsim.core.utils.stage import add_reference_to_stage


class SpawnerTemplate(ABC):
    def __init__(self, catalog, sampler, num_objects):
        self.catalog = catalog
        self.sampler = sampler
        self.spawned_object_paths = []
        self.spawned_object_info = []
        self.num_objects = num_objects
    
    def spawn_random(self):
        names = self._choose_names()
        positions = self.sampler.sample(self.num_objects)
        self._postprocess(names, positions)
        self._spawn(names, positions)

    def clear_objects(self):
        for prim_path in self.spawned_object_paths:
            delete_prim(Sdf.Path(prim_path))
        self.spawned_object_paths.clear()
        self.spawned_object_info.clear()

    def get_spawned_object_info(self):
        return self.spawned_object_info

    @abstractmethod
    def _choose_names(self):
        pass

    @abstractmethod
    def _postprocess(self, names, positions):
        pass

    def _spawn(self, names, positions):
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
        apply_transform(parent_path, position, orientation)
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
    
    def _append_info(self, name, position):
        self.spawned_object_info.append({
            "name": name,
            "position": position,
        })
