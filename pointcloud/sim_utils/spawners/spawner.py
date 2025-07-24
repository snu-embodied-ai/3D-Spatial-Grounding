from abc import ABC, abstractmethod
from sim_utils.assets.asset import AssetCatalog
from sim_utils.samplers import PositionRandomSampler


class Spawner(ABC):
    def __init__(self, catalog: AssetCatalog, sampler: PositionRandomSampler, num_of_objects):
        self.catalog = catalog
        self.sampler = sampler
        self.spawned_object_paths = []
        self.spawned_object_info = []
        self.num_of_objects = num_of_objects
    
    @abstractmethod
    def spawn(self):
        pass

    @abstractmethod
    def clear_objects(self):
        pass

    @abstractmethod
    def get_spawned_object_info(self):
        pass
