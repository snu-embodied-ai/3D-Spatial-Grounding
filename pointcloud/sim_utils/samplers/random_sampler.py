import numpy as np
import random
from . import register
from .sampler import PositionSampler


SPAWN_POSITIONS = [[[x, y, 0.0] for y in np.arange(-0.25, 0.35, 0.1)] for x in np.arange(-0.45, 0.55, 0.1)]


@register("random")
class PositionRandomSampler(PositionSampler):
    def __init__(self, grid=SPAWN_POSITIONS):
        self.grid = grid
    
    def sample(self, num_objects):
        selected_indices = self._sample_indices(num_objects)
        positions = [self.grid[i][j] for (i, j) in selected_indices]
        positions = [self._jiggle(position) for position in positions]
        return positions
    
    def _sample_indices(self, num_objects):
        indices = [
            (x, y)
            for x in range(0, len(self.grid))
            for y in range(0, len(self.grid[0]))
        ]
        selected_indices = []

        for i in range(num_objects):
            if not indices:
                break
            selected_index = random.choice(indices)
            selected_indices.append(selected_index)
            neighbors = self._get_neighbors(selected_index)
            indices = self._exclude_neighbors(indices, neighbors)

        return selected_indices

    def _get_neighbors(self, selected_index):
        x, y = selected_index
        neighbors = [
            (i, j)
            for i in range(x-1, x+2)
            for j in range(y-1, y+2)
            if 0 <= i < len(self.grid) and 0 <= j < len(self.grid)
        ]
        return neighbors

    def _exclude_neighbors(self, indices, neighbors):
        filtered_indices = [index for index in indices if index not in neighbors]
        return filtered_indices

    def _jiggle(self, position, std=0.03):
        position = np.array(position)
        noise = np.random.normal(loc=0.0, scale=std, size=2)
        position[:2] += noise
        return position.tolist()
