import abc
import io

import cv2
import numpy as np
from matplotlib import pyplot as plt

from pytorch_debug_tools.utils.Resolution import Resolution
from pytorch_debug_tools.visualizers.BaseVisualizer import BaseVisualizer


class MatplotlibVisualizer(BaseVisualizer):
    def __init__(self, resolution: Resolution):
        super().__init__(resolution)
        self.resolution = resolution

    @abc.abstractmethod
    def visualize(self, weight: np.ndarray, name: str, num_bins: int) -> np.ndarray:
        figure = self.__create_histogram_figure(weight, name, num_bins)
        np_image = self.__plt_figure_to_np_array(figure)

        # do cleanup
        self.__cleanup()

        return np_image

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
    def __convert_matplotlib_figure_to_np_array(fig: plt.Figure, dpi: int = 180):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)

        return img

    def __plt_figure_to_np_array(self, figure: plt.Figure) -> np.ndarray:
        np_image = self.__convert_matplotlib_figure_to_np_array(figure)
        # resize image if needed
        np_image = self.__resize_image_to_target_resolution(np_image)

        return np_image

    def __resize_image_to_target_resolution(self, np_image: np.ndarray) -> np.ndarray:
        if np_image.shape[0] != self.resolution.height or np_image.shape[1] != self.resolution.width:
            if np_image.shape[0] > self.resolution.height:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_LINEAR
            np_image = cv2.resize(np_image, (self.resolution.width, self.resolution.height),
                                  interpolation=interpolation)

        return np_image

    @staticmethod
    def __cleanup():
        plt.clf()
        plt.close()
