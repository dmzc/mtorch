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
