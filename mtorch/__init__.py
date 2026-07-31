from mtorch.interfaces import ITensor
from mtorch.tensor import Tensor
import mtorch.operator as F
from mtorch.render import render
from mtorch.nn import Module, Sequential, Linear, Sigmoid, MeanSquareLoss
from mtorch.optim import SGD

__version__ = "0.0.13"


def _setup():
    Tensor.__add__ = F.add
    Tensor.__radd__ = F.add
    Tensor.__mul__ = F.mul
    Tensor.__rmul__ = F.mul
    Tensor.__neg__ = F.neg
    Tensor.__sub__ = F.sub
    Tensor.__rsub__ = F.rsub
    Tensor.__truediv__ = F.div
    Tensor.__rtruediv__ = F.rdiv
    Tensor.__pow__ = F.pow


_setup()
__all__ = ["ITensor", "Tensor", "F", "render"]
