from __future__ import annotations
import numpy as np
from mtorch import Tensor, ITensor


class Object:
    _data: np.ndarray

    def __init__(self, data: np.ndarray):
        self._data = data

    def __getitem__(obj: Object, slices: tuple[int | any]):
        print(f"{slices}")

    def __setitem__(self, key, value):
        self._data[key] = value
        print(key)
        print(value)


obj = Object(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]))
# obj[0]  # __get_item__(obj, 0, None)
# obj[0, 1]  # __getitem__(obj, (0,1), None)
# obj[:, 1]  # __getitem__(obj, (slice(None, None, None), 1))
# obj[1:3, 1]  # __getitem__(obj, (slice(1, 3, None), 1))
# obj[1:3, 0:1]  # __getitem__(obj, (slice(1, 3, None), slice(0, 1, None)))
# obj[:2:, 1]  # __getitem__(obj,(slice(None, 2, None), 1))
# obj[1:, 1]  # (slice(1, None, None), 1)
# obj[:2, 1]  # (slice(None, 2, None), 1)
# obj[:2:4]  # slice(None, 2, 4)
# obj[1:4:1]  # slice(1, 4, 1)
# obj[0:2] = [[34, 45, 56], [78, 89, 999]]
# print(obj._data)
# fmt:off
tensor1 = Tensor([
    [1, 2, 3], 
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

tensor2=tensor1[2,1]
print(tensor2)
