from .module import Module
import mtorch.operator as F
from mtorch.interfaces import ITensor

# 激活层


class Sigmoid(Module):
    def forward(self, x) -> ITensor:
        return F.sigmoid(x)
