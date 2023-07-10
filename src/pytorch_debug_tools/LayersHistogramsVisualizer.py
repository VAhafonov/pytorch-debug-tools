from typing import Dict, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn

from pytorch_debug_tools.utils import get_img_from_fig


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

    @classmethod
    def from_tuple(cls, resolution_tuple: Tuple[int, int]):  # return Resolution
        assert len(resolution_tuple) == 2
        return Resolution(height=resolution_tuple[0], width=resolution_tuple[1])


class SingleLayerVisualizer:
    @staticmethod
    def __preprocess_weight(weight: np.ndarray) -> np.ndarray:
        if weight is None:
            raise ValueError("Weight is none")

        if len(weight.shape) > 1:
            # flatten weight
            weight = np.reshape(weight, -1)

        return weight

    @staticmethod
    def __create_histogram_figure(weight: np.ndarray, name: str, num_bins: int) -> plt.Figure:
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.hist(weight, num_bins)
        ax.set_xlabel('Weight value')
        ax.set_ylabel('Quantity')
        ax.set_title(name)

        return fig

    @staticmethod
    def __plt_figure_to_np_array(figure: plt.Figure, resolution: Resolution) -> np.ndarray:
        np_image = get_img_from_fig(figure)
        # resize image if needed
        if np_image.shape[0] != resolution.height or np_image.shape[1] != resolution.width:
            if np_image.shape[0] > resolution.height:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_LINEAR
            np_image = cv2.resize(np_image, (resolution.width, resolution.height), interpolation=interpolation)

        return np_image

    @staticmethod
    def __cleanup():
        plt.clf()
        plt.close()

    @classmethod
    def visualize(cls, weight: np.ndarray, name: str, resolution: Resolution, num_bins: int) -> np.ndarray:
        weight = cls.__preprocess_weight(weight)
        figure = cls.__create_histogram_figure(weight, name, num_bins)
        histogram_image = cls.__plt_figure_to_np_array(figure, resolution)

        cls.__cleanup()

        return histogram_image


class LayersHistogramsVisualizer:
    def __init__(self, nn_model: torch.nn.Module, num_workers: int = 1):
        self.nn_model = nn_model
        self.num_workers = num_workers

    def create_histograms(self) -> Dict[str, np.ndarray]:
        pass

    def create_and_dump_histograms(self, out_dir: str):
        pass
