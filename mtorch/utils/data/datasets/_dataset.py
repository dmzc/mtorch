from mtorch._interfaces import IDataset, ITransform, DatasetData
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib as matplotlib
from collections.abc import Callable


class Dataset(IDataset):
    r"""
    TODO:
        1. 一个数据集拆分成训练、验证、测试数据集

    """

    _data: np.ndarray
    _label: np.ndarray
    _data_transform: ITransform
    _label_transform: ITransform

    def __init__(
        self,
        data_transform: ITransform = None,
        label_transform: ITransform = None,
    ):
        super().__init__()
        self._data = None
        self._label = None
        self._data_transform = data_transform
        self._label_transform = label_transform

    def __len__(self):
        self.__init_data()
        return len(self._data)

    def __getitem__(self, slices) -> DatasetData:
        self.__init_data()
        r"""
        不支持多维切片，而且总是返回数组，让接口一致
        """
        if isinstance(slices, tuple) and len(slices) > 1:  # 只支持第一维度切片
            slices = slices[0]
        if isinstance(slices, int):  # 单个数字直接返回了值，统一下
            slices = slice(slices, slices + 1)

        if self._label is not None:
            return DatasetData(data=self._data[slices], label=self._label[slices])
        return DatasetData(data=self._data[slices])

    def __init_data(self) -> None:
        if self._data is None:
            datas = self.load_data()
            if datas is None:
                raise RuntimeError("There has no data.")
            if isinstance(datas, np.ndarray):
                self._data = datas
            elif isinstance(datas, tuple):
                count = len(datas)
                if count == 0:
                    raise RuntimeError("There has no data.")
                self._data = datas[0]
                if len(datas) >= 2:
                    self._label = datas[1]

    def load_data(self) -> tuple[np.ndarray] | np.ndarray:
        r"""
        子类必须实现此接口加载数据。

        数据加载到_data，标签数据加载到_label
        """
        raise f"Subclass of Dataset must implements load_data"

    def graph(self, title: str = None, imdiately_show=True):
        r"""
        子类不要重写此方法，可视化请重写_graph
        """
        self.__init_data()
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
        plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
        self._graph()
        # 自动调整画布内边距
        plt.tight_layout()
        if imdiately_show:
            plt.show()

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
        super().__init__()
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


class IterableDataset(IDataset):
    def __len__(self):
        return super().__len__()

    def __getitem__(self, slices):
        return super().__getitem__(slices)
