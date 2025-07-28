from . import register
from .spawner import SpawnerTemplate


@register("unique")
class UniqueSpawner(SpawnerTemplate):
    def _choose_names(self):
        names = self.catalog.random_names(self.num_objects)
        return names
    
    def _postprocess(self, names, positions):
        return super()._postprocess(names, positions)
    