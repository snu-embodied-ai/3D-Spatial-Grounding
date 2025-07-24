import numpy as np
import random
from .sampler import PositionSampler


SPAWN_POSITIONS = [[[x, y, 0.0] for y in np.arange(-0.25, 0.35, 0.25)] for x in np.arange(-0.45, 0.55, 0.225)]
NUM_ALIGNED_LINES = 2


class PositionAlignedSampler(PositionSampler):
    def __init__(self, grid=SPAWN_POSITIONS):
        self.grid = grid
    
    def sample(self, num_objects):
        selected_indices = self._sample_positions(num_objects)
        positions = [self.grid[i][j] for (i, j) in selected_indices]
        positions = [self._jiggle(position) for position in positions]
        return positions
    
    def _sample_positions(self, num_objects):
        num_lines = len(self.grid[0])
        selected_lines = random.sample(range(num_lines), NUM_ALIGNED_LINES)
        num_samples_per_line = self._get_num_indices_per_line(num_objects)

        selected_indices = []
        for i in range(len(selected_lines)):
            y = selected_lines[i]
            selected_xs = random.sample(range(len(self.grid)), num_samples_per_line[i])
            for x in selected_xs:
                selected_indices.append((x, y))

        return selected_indices

    def _get_num_indices_per_line(self, num_objects):
        num_samples_per_line = [num_objects // NUM_ALIGNED_LINES] * NUM_ALIGNED_LINES
        remainders = random.sample(range(NUM_ALIGNED_LINES), num_objects % NUM_ALIGNED_LINES)
        remainders_per_line = [1 if i in remainders else 0 for i in range(NUM_ALIGNED_LINES)]
        for i in range(NUM_ALIGNED_LINES):
            num_samples_per_line[i] += remainders_per_line[i]
        return num_samples_per_line

    def _jiggle(self, position, std=0.02):
        position = np.array(position)
        noise = np.random.normal(loc=0.0, scale=std, size=2)
        position[:2] += noise
        return position.tolist()
