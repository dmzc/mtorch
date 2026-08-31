from __future__ import annotations

from mtorch._interfaces import ITensor, IOperator
from ._core import _data_array, _isscalar, _ones_like
from mtorch.autograd import backward
from typing import Any

ENABLE_BACKPROGATION = True  # TODO:去掉


class Tensor(ITensor):
    __name: str

    # TODO：修改为__array_func__

    # 貌似不需要这个，numpy内部实现__add__、__radd__方法应该做了类型判断，
    # 遇到不识别的类型，会自动转交。

    __array_priority__ = 200

    _require_grad: bool

    def __init__(
        self,
        data: Any,  # 数值、列表、元组、DataArray，不能是Tensor实例
        creator: IOperator = None,
        name: str = None,
        require_grad: bool = False,
    ):
        if _isscalar(data) or isinstance(data, list) or isinstance(data, tuple):
            data = _data_array(data)
        self.data = data
        self.grad = None
        self.__name = name
        self._require_grad = require_grad
        if ENABLE_BACKPROGATION:
            creator = self.creator = creator
            if creator is None:
                self.generation = 0
            else:
                self.generation = creator.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self) -> None:
        backward(self)

    @property
    def id(self) -> str:
        return f"_{id(self)}_"

    @property
    def name(self) -> str:
        ret_name = ""
        if self.__name is not None:
            ret_name = f"{self.__name}"
            if ENABLE_BACKPROGATION:
                ret_name = f"{ret_name}({self.generation})\n数据：{self.data}"
                if self.grad is not None:
                    ret_name = f"{ret_name}\n梯度：{self.grad}"
            else:
                ret_name = f"{ret_name}\n数据：{self.data}"
        else:
            if ENABLE_BACKPROGATION:
                ret_name = f"数据：{self.data}\n层级：{self.generation}"
                if self.grad is not None:
                    ret_name = f"{ret_name}\n梯度：{self.grad}"
            else:
                ret_name = f"数据：{self.data}\n"
        return ret_name

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def require_grad(self):
        return self._require_grad

    @require_grad.setter
    def require_grad(self, value):
        # 先不做类型校验，看调用次数是否频繁
        self._require_grad = value

    def init_grad(self):
        self.grad = _ones_like(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        data_str = str(self.data).replace("\n", "\n" + " " * 9)
        return f"Tensor( {data_str} )"


class Parameter(Tensor):

    @property
    def require_grad(self):
        return True

    @require_grad.setter
    def require_grad(self, value):
        raise RuntimeError("Parameter's require_grad always True!")

    def __repr__(self):
        data_str = str(self.data).replace("\n", "\n" + " " * 9)
        return f"Parameter( {data_str} )"
