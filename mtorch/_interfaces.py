from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Protocol
from dataclasses import dataclass
from typing import Literal, TypeAlias, Any, Optional
import numpy as np  # TODO:删除，上层用DataArray

DataArray: TypeAlias = np.ndarray[Any, Any]


class ITensor(ABC):
    """
    张量
    """

    data: DataArray
    creator: Optional[IOperator]
    generation: int
    require_grad: bool
    grad: Optional[DataArray]

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


class IOperator(Protocol):
    """
    算子
    """

    inputs: list[ITensor]
    outputs: list[weakref.ref[ITensor]]
    label: str
    generation: int  # TODO:deperated

    def backward(self, dout: DataArray) -> DataArray: ...

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...


class IModule(ABC):
    @abstractmethod
    def forward(self, x: Any) -> Any: ...

    @abstractmethod
    def parameters(self) -> Iterable[ITensor]: ...


class IOptimizer(ABC):
    _params_obj: IModule

    def __init__(self, params_obj: IModule):
        super().__init__()
        self._params_obj = params_obj

    @abstractmethod
    def step(self): ...


Slice_Type: TypeAlias = int | list[int] | slice | tuple[int | slice]

Dataset_Type: TypeAlias = Literal["train", "test", "val"]


@dataclass
class DatasetData:
    data: DataArray
    label: DataArray | None = None


class IDataset(ABC):
    def __len__(self) -> int:
        """
        返回数据量
        """
        raise NotImplementedError("Subclasses of IDataset should implement __len__.")

    def __getitem__(self, slices: Slice_Type) -> Any:
        r"""
        根据切片返回对应数据，但是这个切片只对条数起作用。
        """
        raise NotImplementedError(
            "Subclasses of IDataset should implement __getitem__."
        )


class IDataLoader(ABC):
    r"""
    数据加载器。

    注意：返回的数据是原始数组的视图，与内部数据集共享内存缓冲区。
    外部直接修改返回数组的元素，会污染数据集内部原始数据。
    若需要对返回数据做写操作，务必先调用 `.copy()` 生成独立副本。
    TODO:
    """

    _dataset: IDataset

    def __iter__(self) -> Any:
        """
        返回迭代器对象，和__next__配合使用
        """
        raise NotImplementedError(
            "Subclasses of IDataLoader should implement __iter__."
        )

    def __next__(self) -> tuple[DataArray] | DataArray:
        """
        迭代器对象返回下一条或下一批数据
        """
        raise NotImplementedError(
            "Subclasses of IDataLoader should implement __next__."
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
    _params: Any  # 超参数

    def train(self) -> None:
        pass

    def evaluation(self) -> None:
        pass


class ITransform(Protocol):
    r"""
    数据变换接口。

    可调用实例，执行样本预处理逻辑。
    输入可能是数据集原始数组的内存视图，**禁止对入参做原地修改**，
    实现必须返回变换后的独立数组副本，避免污染底层原始数据。

    Notes:
        - 仅支持 numpy.ndarray 作为输入输出
        - __call__ 接收单样本数组，返回处理完成的样本数组
    """

    # def __repr__(self): ...

    def __repr__(self) -> str: ...

    def __call__(self, *args: Any, **kwds: Any): ...
