from mtorch.interfaces import IDataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib as matplotlib
from collections.abc import Callable


class Dataset(IDataset):
    r"""
    TODO:
        1. 一个数据集拆分成训练、验证、测试数据集
        2. 大数据数据集懒加载

    """

    _enable_label: bool
    _data: np.ndarray
    _label: np.ndarray

    def __init__(self, enable_label: bool = False):
        super().__init__()
        self._enable_label = enable_label
        self._data = None
        self._label = None
        self._external_axes = None

    def __len__(self):
        self.__init_data()
        return len(self._data)

    def __getitem__(self, slices) -> tuple[np.ndarray] | np.ndarray:
        self.__init_data()
        if self._enable_label:
            return (self._data[slices], self._label[slices])
        return self._data[slices]

    def __init_data(self) -> None:
        if self._data is None:
            datas = self.load_data()
            enable_label = self._enable_label
            if datas is None:
                raise RuntimeError("There has no data.")
            if isinstance(datas, np.ndarray):
                if enable_label:
                    raise ValueError("Enable_label is True,label must be returned.")
                self._data = datas
            elif isinstance(datas, tuple):
                if enable_label and len(datas) < 2:
                    raise ValueError("Enable_label is True,label must be returned.")
                self._data = datas[0]
                if enable_label:
                    self._label = datas[1]

    def load_data(self) -> tuple[np.ndarray] | np.ndarray:
        r"""
        子类必须实现此接口加载数据。

        数据加载到_data，标签数据加载到
        `_enable_label` 为`True`时，必须加载标签数据。
        """
        raise f"Subclass of Dataset must implements fetch_data"

    def graph(self, title: str = None, imdiately_show=True, axes: Axes = None):
        r"""
        子类不要重写此方法，可视化请重写_graph
        """
        self._external_axes = axes
        self.__init_data()
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
        plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
        self._graph()
        # 自动调整画布内边距
        plt.tight_layout()
        if imdiately_show:
            plt.show()
        self._external_axes = None

    def _graph(self) -> None:
        r"""
        子类可以实现此接口以可视化的形式展示此数据，默认显示“xxx数据集没有实现可
        视化方法”。

        如果使用plt，此方法不要调用show，graph会根据需要决定是否调用plt.show
        """

        """
        figure:图窗
        axes:绘图对象
        plt: pyplot 在内部维护一个全局状态：记录 “当前激活的 figure、当前激活的axes”。
            plot.plot() 并不显式指定画到哪，它自动画到当前活跃的
            Axes。plot.subplots() 是 pyplot 模块提供的工厂函数，用来创建 Figure
            + Axes。创建完之后，后续绘图全部直接操作ax对象，不再依赖 pyplot 全局
            状态
        """
        axes = self.create_subplot()
        # 清除坐标轴刻度
        axes.set_xticks([])
        axes.set_yticks([])

        # 在画布中心放置文本
        axes.text(
            0.5,
            0.5,
            f"{self.__class__.__name__}不支持可视化",
            fontsize=14,
            ha="center",
            va="center",
            transform=axes.transAxes,
        )
        # 移除边框
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.spines["bottom"].set_visible(False)
        axes.spines["left"].set_visible(False)

    def create_subplot(
        self,
        figsize: tuple[int] = (6, 4),
        show_toolbar: bool = False,
        title: str = None,
    ) -> Axes:
        if title is None:
            title = " "
        figure, axes = plt.subplots(figsize=figsize)
        figure.canvas.manager.set_window_title(title)
        if not show_toolbar:
            figure.canvas.manager.toolbar.setVisible(False)  # 去掉toolbar显示
        return axes


class FunctionDataset(Dataset):
    r"""
    函数数据集
    """

    _func: Callable
    _data_size: int

    def __init__(self, func, data_size: int = 100):
        super().__init__(enable_label=True)
        self._func = func
        self._data_size = data_size


class UnivariateFunctionDataset(FunctionDataset):
    r"""
    一元函数数据集。

    参数为np.ndarray
    """

    def load_data(self):
        data_size = self._data_size
        x: np.ndarray = np.random.rand(data_size, 1)
        y = self._func(x) + np.random.rand(data_size, 1)
        return (x, y)

    def _graph(self, title: str = None):
        axes = self.create_subplot(title=title)
        # 这里为了画出正弦图像，要用连续的自变量，而x随机生成的，所以不满足要求
        x = np.arange(0, 1, 0.01)[:, np.newaxis]
        y = self._func(x)
        axes.scatter(x, y)


class SprialDataset(Dataset):
    r"""
    螺旋数据集（三类）
    """

    _data_size_per_category: int

    def __init__(self, data_size_per_category: int = 100):
        super().__init__(enable_label=True)
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
