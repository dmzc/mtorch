from mtorch.utils.data.transforms import Flatten
import numpy as np


def test_flatten():
    arr = np.array([[1, 2], [3, 4]])
    arr = Flatten()(arr)
    assert arr.tolist() == [1, 2, 3, 4] and str(Flatten()) == "扁平化", "扁平化变换成功"
