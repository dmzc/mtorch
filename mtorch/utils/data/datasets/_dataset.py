from __future__ import annotations
from collections.abc import Callable

import numpy as np

from mtorch._interfaces import DatasetData, IDataset, ITransform


class AbstractDataset(IDataset):

    _data_transform: ITransform
    _label_transform: ITransform

    def __init__(
        self,
        data_transform: ITransform = None,
        label_transform: ITransform = None,
    ):
        super().__init__()
        self._data_transform = data_transform
        self._label_transform = label_transform

    def __getitem__(self, slices) -> DatasetData:
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
        data = self.get_item(slices=slices)
        data_transform = self._data_transform
        label_transform = self._label_transform
        if data_transform is not None or (
            label_transform is not None and data.label is not None
        ):
            if data_transform is not None:
                n_label: list[np.ndarray] = []
                for item in data.data:
                    n_label.append(data_transform(item))
                data.data = np.array(n_label)
            if label_transform is not None:
                n_label: list[np.ndarray] = []
                for item in data.label:
                    n_label.append(label_transform(item))
                data.label = np.array(n_label)
            return data
        else:
            return data

    def __len__(self):
        return self.len()

    def len(self) -> int:
        raise NotImplementedError("Sub class must implements len!")

    def get_item(self, slices) -> DatasetData:
        raise NotImplementedError("Sub class must implements get_item!")


class MemoryDataset(AbstractDataset):
    r"""
    内存数据集，样本全部驻留内存。

    支持两种工作模式：
        1. 构造时直接传入 data / label：数据集直接使用传入的内存数组，无需动态加载。
        2. 构造时不传入数据：子类需要重写 `load_data()`，完成动态加载，加载完成后内部填充数据。

    Notes:
        - data、label 均要求为 numpy.ndarray，样本数量维度必须对齐
        - 两种模式二选一，不可混用
        - 索引、变换流水线、长度统计全部继承上层抽象数据集逻辑
    """

    _data: np.ndarray
    _label: np.ndarray
    _inited: bool

    def __init__(
        self,
        data_transform=None,
        label_transform=None,
        data: np.ndarray | list = None,
        label: np.ndarray | list = None,
    ):
        super().__init__(data_transform, label_transform)
        if data is not None:
            if isinstance(data, list):
                data = np.array(data)
            if label is not None and isinstance(label, list):
                label = np.array(label)
            self._inited = True
            self._data = data
            self._label = label
        else:
            self._inited = False
            self._data = None
            self._label = None

    def len(self):
        self.__init_data()
        return len(self._data)

    def get_item(self, slices):
        self.__init_data()
        if self._label is not None:
            return DatasetData(data=self._data[slices], label=self._label[slices])
        return DatasetData(data=self._data[slices])

    def __init_data(self) -> None:
        if self._inited:
            return
        datas = self.fetch_data()
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
        self._inited = True

    def fetch_data(self) -> tuple[np.ndarray] | np.ndarray:
        r"""
        对于内存数据集，可在初始化阶段直接传入完整数据，此时无需实现该方法。
        采用延迟加载模式的子类，必须实现 `load_data()` 完成数据初始化。
        """
        if not self._inited:
            raise NotImplementedError(
                "Subclass not initialized. Lazy‑loaded datasets must implement load_data(). "
                "For in‑memory datasets, pass fully‑loaded data in constructor instead."
            )


class FunctionDataset(MemoryDataset):
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

    def fetch_data(self):
        data_size = self._data_size
        x: np.ndarray = self._x_data
        if x is None:
            x = np.random.rand(data_size, 1)
        y = self._func(x) + np.random.rand(data_size, 1)
        return (x, y)
