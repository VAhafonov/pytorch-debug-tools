import abc

import numpy as np

from pytorch_debug_tools.utils.Resolution import Resolution


class BaseVisualizer(abc.ABC):
    def __init__(self, resolution: Resolution):
        self.resolution = resolution

    @abc.abstractmethod
    def visualize(self, weight: np.ndarray, name: str, resolution: Resolution, num_bins: int) -> np.ndarray:
        raise NotImplementedError
