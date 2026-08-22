from .module import Module
import mtorch.core.operator as F
from mtorch.interfaces import ITensor

# 激活层


class Sigmoid(Module):
    def forward(self, x) -> ITensor:
        return F.sigmoid(x)
