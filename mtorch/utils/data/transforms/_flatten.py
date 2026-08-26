from mtorch._interfaces import ITransform
import numpy as np

r"""扁平化
将一个数组进行扁平化处理
"""


class Flatten(ITransform):

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x
        return x.flatten()

    def __repr__(self):
        return "扁平化"
