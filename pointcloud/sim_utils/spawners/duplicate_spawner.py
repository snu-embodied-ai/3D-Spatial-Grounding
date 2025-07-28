from . import register
from .spawner import SpawnerTemplate


@register("duplicate")
class DuplicateSpawner(SpawnerTemplate):
    def _choose_names(self):
        names = self.catalog.random_duplicate_names(self.num_objects)
        return names
    
    def _postprocess(self, names, positions):
        return super()._postprocess(names, positions)
