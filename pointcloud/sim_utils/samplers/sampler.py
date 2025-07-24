from abc import ABC, abstractmethod
import numpy as np


SPAWN_POSITIONS = [[[x, y, 0.0] for y in np.arange(-0.35, 0.45, 0.1)] for x in np.arange(-0.35, 0.45, 0.1)]


class PositionSampler(ABC):
    def __init__(self, grid=SPAWN_POSITIONS):
        self.grid = grid
    
    @abstractmethod
    def sample(self, num_objects):
        pass
