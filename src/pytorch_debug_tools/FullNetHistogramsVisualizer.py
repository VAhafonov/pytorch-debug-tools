from typing import Dict

import numpy as np


class FullNetHistogramsVisualizer:
    def __init__(self, num_workers: int = 1):
        # currently only singlethread version is supported
        assert num_workers == 1
        self.num_workers = num_workers

    def create_histograms(self) -> Dict[str, np.ndarray]:
        pass

    def create_and_dump_histograms(self, out_dir: str):
        pass
