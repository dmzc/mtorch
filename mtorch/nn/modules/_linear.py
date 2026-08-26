from mtorch.nn.modules._module import Module
from mtorch._interfaces import ITensor
import mtorch.core.operator as F
from mtorch.core._tensor import Tensor
import numpy as np


# 线性层
class Linear(Module):
    _w: np.ndarray
    _b: np.ndarray

    def __init__(self, input_size: int, hidden_size: int, use_bias=True):
        super().__init__()
        # TODO:参数初始化方式
        W_data = np.random.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(
            1 / input_size
        )
        # self.W.data = W_data
        self._w = Tensor(W_data, require_grad=True)
        # self._w = Tensor(np.random.randn(input_size, hidden_size), require_grad=True)
        if use_bias:
            self._b = Tensor(
                np.random.randn(hidden_size).astype(np.float32), require_grad=True
            )
        else:
            self._b = None

    def forward(self, x) -> ITensor:
        return F.linear(x, self._w, self._b)
