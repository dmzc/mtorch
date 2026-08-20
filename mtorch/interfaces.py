from __future__ import annotations
import numpy as np
import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TypeAlias


class ITensor(ABC):
    """
    张量
    """

    data: np.ndarray
    creator: IOperator
    generation: int
    require_grad: bool
    grad: ITensor

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def shape(self): ...

    @property
    def ndim(self): ...

    @property
    def size(self): ...

    @property
    def dtype(self): ...

    @abstractmethod
    def init_grad(self) -> None: ...

    def clear_grad(self) -> None: ...

    def backward(self) -> None: ...


class IOperator(ABC):
    """
    算子
    """

    inputs: list[ITensor]
    outputs: list[weakref.ref[ITensor]]
    label: str
    generation: int

    def forward(self, *xs: np.ndarray) -> any: ...

    def backward(self, dout: ITensor) -> ITensor: ...

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...


class IModule(ABC):

    @abstractmethod
    def forward(self, x: any) -> any: ...

    @abstractmethod
    def params(self) -> Iterable[ITensor]: ...


class IOptimizer(ABC):

    _params_obj: IModule

    def __init__(self, params_obj: IModule):
        super().__init__()
        self._params_obj = params_obj

    @abstractmethod
    def step(self): ...


ISliceType: TypeAlias = int | slice | tuple[int | slice]


class IDataset(ABC):

    def __len__(self) -> int:
        """
        返回数据量
        """
        raise NotImplementedError(f"Subclasses of IDataset should implement __len__.")

    def __getitem__(self, slices: ISliceType):
        """
        根据切片返回对应数据。
        """
        raise NotImplementedError(
            f"Subclasses of IDataset should implement __getitem__."
        )


class IDataLoader(ABC):
    """
    数据集
    """

    _dataset: IDataset

    def __iter__(self):
        """
        返回迭代器对象，和__next__配合使用
        """
        raise NotImplementedError(
            f"Subclasses of IDataLoader should implement __iter__."
        )

    def __next__(self) -> tuple[np.ndarray] | np.ndarray:
        """
        迭代器对象返回下一条或下一批数据
        """
        raise NotImplementedError(
            f"Subclasses of IDataLoader should implement __next__."
        )


class ITrainer(ABC):
    r"""
    TODO:
        1. 训练
        2. 评估，不在这里做，应该是抽象工具方法，但训练过程也有精度、损失评估
    """

    _model: IModule  # 模型
    _dataloader: IDataLoader  # 数据加载器
    _optimizer: IOptimizer  # 梯度更新器
    _params: any  # 超参数

    def train(self) -> None:
        pass

    def evaluation(self) -> None:
        pass
