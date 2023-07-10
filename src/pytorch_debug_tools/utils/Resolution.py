from typing import Tuple


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