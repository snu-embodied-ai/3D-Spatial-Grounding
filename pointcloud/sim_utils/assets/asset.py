import random
from dataclasses import dataclass
from typing import Dict

from .asset_list import ASSET_NAMES, ASSET_NAME_TO_PATH, ASSET_NAME_TO_SPAWN_HEIGHT


LOCAL_ASSET_ROOT_PATH = "/home/choij/isaac-sim/pointcloud/assets"


@dataclass
class Asset:
    name: str
    usd_path: str
    spawn_height: float


class AssetCatalog:
    def __init__(self, isaac_root, local_root=LOCAL_ASSET_ROOT_PATH, asset_list=ASSET_NAMES):
        self.assets: Dict[str, Asset] = {}
        self.isaac_root = isaac_root
        self.local_root = local_root

        for name in asset_list:
            relative_path = ASSET_NAME_TO_PATH[name]
            spawn_height = ASSET_NAME_TO_SPAWN_HEIGHT[name]

            full_path = self._get_full_path(relative_path)

            self.assets[name] = Asset(
                name=name,
                usd_path=full_path,
                spawn_height=spawn_height
            )
    
    def _get_full_path(self, relative_path):
        if relative_path.startswith("/Isaac"):
            root_path = self.isaac_root
        else:
            root_path = self.local_root
        return root_path + relative_path
    
    def get_path(self, name):
        return self.assets[name].usd_path

    def get_spawn_height(self, name):
        return self.assets[name].spawn_height

    def random_names(self, count):
        return random.sample(list(self.assets.keys()), count)

    def random_duplicate_names(self, count):
        if (count < 2):
            return
        names = random.choices(list(self.assets.keys()), k=count-1)
        names += [random.choice(list(set(names)))]
        return names
