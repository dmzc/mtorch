import numpy as np

from ._dataset import MemoryDataset

r"""
    螺旋数据集（三类）
"""


class Sprial(MemoryDataset):

    _data_size_per_category: int

    def __init__(self, data_size_per_category: int = 100):
        super().__init__()
        self._data_size_per_category = data_size_per_category

    def fetch_data(self):
        data_size_per_category = self._data_size_per_category
        category_size = 3
        input_dim = 2
        data_size = category_size * data_size_per_category
        x = np.zeros((data_size, input_dim), dtype=np.float32)
        t = np.zeros(data_size, dtype=np.int8)

        for category in range(category_size):
            for idx in range(data_size_per_category):
                rate = idx / data_size_per_category
                radius = 1.0 * rate  # 半径
                theta = category * 4.0 + 4.0 * rate + np.random.randn() * 0.2  # 极叫
                data = np.array([radius * np.sin(theta), radius * np.cos(theta)])
                data_idx = data_size_per_category * category + idx
                x[data_idx] = data
                t[data_idx] = category
        return (x, t)

    def get_category_info(self) -> tuple[list[str], list[str]]:
        return (["o", "x", "^"], ["orange", "blue", "green"])
