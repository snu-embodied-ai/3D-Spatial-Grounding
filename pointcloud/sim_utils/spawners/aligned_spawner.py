from . import register
from .spawner import SpawnerTemplate


THRESHOLD = 0.07


@register("aligned")
class AlignedSpawner(SpawnerTemplate):
    def __init__(self, catalog, sampler, num_of_objects, target_object):
        super().__init__(catalog, sampler, num_of_objects)
        self.target_object = target_object

    def _choose_names(self):
        names = self.catalog.random_names(self.num_objects)
        return names
    
    def _postprocess(self, names, positions):
        aligned_target_line = positions[0][1]
        num_targets = 0
        for position in positions:
            if abs(position[1] - aligned_target_line) < THRESHOLD:
                num_targets += 1
            else:
                break

        names[:num_targets] = [self.target_object] * num_targets
    