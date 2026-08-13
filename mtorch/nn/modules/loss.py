from mtorch.nn.modules import Module
from mtorch.interfaces import ITensor
import mtorch.operator as F
import numpy as np


class MeanSquareLoss(Module):
    def forward(self, y_actual: np.ndarray, y_expect: np.ndarray) -> ITensor:
        return F.mean_square_loss(y_actual=y_actual, y_expect=y_expect)


class CrossEntroyLoss(Module):

    _axis: int

    def __init__(self, axis: int):
        super().__init__()
        self._axis = axis

    def forward(self, x: np.ndarray, t: np.ndarray):
        return F.CrossEntropyLoss(x, t, axis=self._axis)
