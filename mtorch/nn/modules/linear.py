from mtorch.nn.modules.module import Module
from mtorch.interfaces import ITensor
import mtorch.operator as F
from mtorch.tensor import Tensor
import numpy as np


# 线性层
class Linear(Module):
    _w: np.ndarray
    _b: np.ndarray

    def __init__(self, input_size: int, hidden_size: int, use_bias=True):
        super().__init__()
        # TODO:参数初始化方式
        self._w = Tensor(np.random.randn(input_size, hidden_size), require_grad=True)
        self._b = Tensor(np.random.randn(hidden_size), require_grad=True)

    def forward(self, x) -> ITensor:
        return F.linear(x, self._w, self._b)
