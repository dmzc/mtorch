from mtorch.nn.modules.module import Module
from mtorch.interfaces import ITensor
import mtorch.operator as F


class Softmax(Module):
    _axis: int

    def __init__(self, axis: int):
        super().__init__()
        self._axis = axis

    def forward(self, x) -> ITensor:
        return F.softmax(x, self._axis)


class LogSoftmax(Softmax):

    def forward(self, x):
        return F.logSoftmax(x, self._axis)
