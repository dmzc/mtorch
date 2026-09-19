from mtorch.typing import IModule, ITensor
from collections.abc import Iterable
from typing import Any
from mtorch.nn.parameter import Parameter


class Module(IModule):

    __subs: set[str]

    def __init__(self):
        self.__subs = set()

    def __setattr__(self, name, value):
        if isinstance(value, ITensor):
            # 中间变量上记录的梯度也需要清除
            self.__subs.add(name)
        if isinstance(value, IModule):
            self.__subs.add(name)
        super().__setattr__(name, value)

    def parameters(self) -> Iterable[Parameter]:
        subs = self.__subs
        if subs is None:
            return
        for sub in subs:
            obj = getattr(self, sub)
            if isinstance(obj, Parameter):
                yield obj
            else:
                m: IModule = obj
                yield from m.parameters()

    def tensors(self) -> Iterable[ITensor]:
        subs = self.__subs
        if subs is None:
            return
        for sub in subs:
            obj = getattr(self, sub)
            if isinstance(obj, ITensor):
                yield obj
            else:
                m: IModule = obj
                yield from m.tensors()

    def clear_grads(self):
        for tensor in self.tensors():
            tensor.clear_grad()

    def __repr__(self) -> str:
        return f"{self.name}()"

    def state_dict(self) -> dict[str, Any]:
        # TODO:
        raise NotImplementedError("Subclass must implements state_dict.")

    def load_state_dict(self, state: dict[str, Any]):
        raise NotImplementedError("Subclass must implements load_dict.")

    @property
    def name(self) -> str:
        return self.__class__.__name__
