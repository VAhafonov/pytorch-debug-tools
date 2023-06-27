from typing import Dict

import numpy as np
import torch.nn


class Resolution:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    @classmethod
    def from_str(cls, resolution_string: str):  # returns Resolution
        resolution_string = resolution_string.strip()
        separator_idx = resolution_string.find('x')
        if separator_idx == -1:
            raise Exception("Resolution string has wrong format")
        height = int(resolution_string[:separator_idx])
        width = int(resolution_string[separator_idx + 1:])

        return Resolution(height=height, width=width)

class SingleLayerVisualiser:
    def visualise(self, weight: np.ndarray, resolution: Resolution) -> np.ndarray:
        pass


class LayersHistogramsVisualizer:
    pass
