from typing import overload, OrderedDict

from mtorch.nn.modules.module import Module
import torch

torch.nn.Linear


# 串型网络结构
class Sequential(Module):

    subs: tuple[Module, ...]

    # TODO:串行模型层之间，可以提供类似加减乘除的语法糖

    @overload
    def __init__(self, *args: Module) -> None: ...

    @overload
    def __init__(self, arg: OrderedDict[str, Module]) -> None: ...

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            subs: list[Module] = []
            for key, module in args[0].items():
                subs.append(module)
                setattr(self, key, module)
            self.subs = tuple(subs)
        else:
            self.subs = args
            for index, sub in enumerate(args):
                setattr(self, f"{index}", sub)

    def forward(self, x: any) -> any:
        for sub in self.subs:
            x = sub.forward(x)
        return x
