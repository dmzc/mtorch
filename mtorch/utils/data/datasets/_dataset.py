from collections.abc import Callable

import numpy as np

from mtorch._interfaces import DatasetData, IDataset, ITransform


class Dataset(IDataset):

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
        if isinstance(slices, tuple):  # 只支持第一维度切片
            if len(slices) == 1:
                slices = slices[0]
            else:
                raise TypeError("Does not support multi‑dimensional slicing!")
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
        raise NotImplementedError("Subclass of Dataset must implements load_data")


class MemoryDataset(Dataset):
    r"""
    内存数据集，数据本来就在内存，传进来数据集做统一接口管理。
    """

    def __init__(
        self,
        data_transform=None,
        label_transform=None,
        data: np.ndarray = None,
        label: np.ndarray = None,
    ):
        super().__init__(data_transform, label_transform)
        self._data = data
        self._label = label


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

    _x_data: np.ndarray

    def __init__(self, func, data_size=100, x_data: np.ndarray = None):
        super().__init__(func, data_size)
        self._x_data = x_data

    def load_data(self):
        data_size = self._data_size
        x: np.ndarray = self._x_data
        if x is None:
            np.random.rand(data_size, 1)
        y = self._func(x) + np.random.rand(data_size, 1)
        return (x, y)
