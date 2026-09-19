from mtorch.nn.modules.module import Module
from mtorch.typing import ITensor
import mtorch.core.operator as F
from mtorch.nn.parameter import Parameter
import numpy as np


# 线性层
class Linear(Module):
    _w: Parameter
    _b: Parameter

    def __init__(self, input_size: int, hidden_size: int, use_bias=True):
        super().__init__()
        # TODO:参数初始化方式
        W_data = np.random.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(
            1 / input_size
        )
        self._w = Parameter(W_data)
        if use_bias:
            self._b = Parameter(np.random.randn(hidden_size).astype(np.float32))
        else:
            self._b = None

    def __repr__(self):
        input_size, hidden_size = self._w.shape
        use_bias = self._b is not None
        return f"{self.name}(input_size = {input_size}, hidden_size = {hidden_size}, use_bias = {use_bias})"

    def forward(self, x) -> ITensor:
        return F.linear(x, self._w, self._b)
