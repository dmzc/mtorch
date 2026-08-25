from ._dataset import Dataset
import numpy as np

r"""
    螺旋数据集（三类）
"""


class Sprial(Dataset):

    _data_size_per_category: int

    def __init__(self, data_size_per_category: int = 100):
        super().__init__()
        self._data_size_per_category = data_size_per_category

    def load_data(self):
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

    def _graph(self, title: str = None):
        if title is None:
            title = "螺旋数据集"
        axes = self.create_subplot(title=title)
        markers, colors = self.get_category_info()
        data = self._data
        label = self._label
        for idx in range(len(data)):
            category = label[idx]
            item: np.ndarray = data[idx]
            axes.scatter(
                item[0],
                item[1],
                s=40,
                marker=markers[category],
                c=colors[category],
            )

    def get_category_info(self) -> tuple[list[str], list[str]]:
        return (["o", "x", "^"], ["orange", "blue", "green"])
