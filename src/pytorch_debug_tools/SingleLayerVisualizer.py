import numpy as np

from pytorch_debug_tools.utils.Resolution import Resolution
from pytorch_debug_tools.visualizers.MatplotlibVisualizer import MatplotlibVisualizer


class SingleLayerVisualizer:
    def __init__(self, resolution: Resolution):
        # TODO remove this hardcode
        self.__inner_visualizer = MatplotlibVisualizer(resolution)

    @staticmethod
    def __preprocess_weight(weight: np.ndarray) -> np.ndarray:
        if weight is None:
            raise ValueError("Weight is none")

        if len(weight.shape) > 1:
            # flatten weight
            weight = np.reshape(weight, -1)

        return weight

    def visualize(self, weight: np.ndarray, name: str, num_bins: int = 50) -> np.ndarray:
        weight = self.__preprocess_weight(weight)
        histogram_image = self.__inner_visualizer.visualize(weight, name, num_bins)

        return histogram_image
