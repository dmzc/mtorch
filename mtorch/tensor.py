from __future__ import annotations
import numpy as np
from mtorch.config import ENABLE_BACKPROGATION
from mtorch.interfaces import ITensor, IOperator


class Tensor(ITensor):

    __name: str

    # TODO：修改为__array_func__

    # 貌似不需要这个，numpy内部实现__add__、__radd__方法应该做了类型判断，遇到不识别的类型，会自动转交。

    # __array_priority__ = 200

    def __init__(
        self,
        data: any,  # 数值、np.ndarray，不能是Variable实例
        creator: IOperator = None,
        name: str = None,
        is_input: bool = False,
        require_grad: bool = False,
    ):
        if np.isscalar(data) or isinstance(data, list) or isinstance(data, tuple):
            data = np.array(data)
        self.data = data
        self.grad = None
        self.__name = name
        self.is_input = is_input
        self.require_grad = require_grad
        if ENABLE_BACKPROGATION:
            creator = self.creator = creator
            if creator is None:
                self.generation = 0
            else:
                self.generation = creator.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self, retain_grad=False) -> None:
        if self.creator is None:
            return

        if self.grad is None:
            self.grad = Tensor(data=np.ones_like(self), is_input=self.is_input)
        creators: list[IOperator] = []
        seen_set: set = set()

        def add_creator(creator):
            if creator not in seen_set:
                seen_set.add(creator)
                creators.append(creator)
                creators.sort(key=lambda x: x.generation)

        add_creator(self.creator)

        while creators:
            creator = creators.pop()
            gys = [output().grad for output in creator.outputs]
            gxs = creator.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(creator.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                x.grad.is_input = x.is_input
                if x.creator is not None:
                    add_creator(x.creator)
            if not retain_grad:
                for y in creator.outputs:
                    y().grad = None

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

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "tensor(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "tensor(" + p + ")"
