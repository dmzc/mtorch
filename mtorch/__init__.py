from mtorch.interfaces import ITensor
from mtorch.tensor import Tensor
import mtorch.operator as F
from mtorch.render import render
from mtorch.nn import (
    Module,
    Sequential,
    Linear,
    Sigmoid,
    MeanSquareLoss,
    CrossEntroyLoss,
    Softmax,
    LogSoftmax,
)
from mtorch.optim import SGD, Adam

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
    Tensor.__matmul__ = F.matmul
    Tensor.__rmatmul__ = F.rmatmul
    Tensor.__imatmul__ = F.imatmul
    Tensor.__getitem__ = F.get_item


_setup()
__all__ = [
    "ITensor",
    "Tensor",
    "F",
    "render",
    "Module",
    "Sequential",
    "Linear",
    "Sigmoid",
    "MeanSquareLoss",
    "SGD",
    "Adam",
]
