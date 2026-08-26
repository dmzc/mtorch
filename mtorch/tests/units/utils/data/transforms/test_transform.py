from mtorch.utils.data.transforms import (
    Compose,
    # Conditional,
    # RandomApply,
    # RandomChoice,
    # RandomOrder,
    # Branch,
    Flatten,
    ToFloat32,
)
import numpy as np


def test_compose():
    arr = np.array([[1, 2], [3, 4]])
    transform = Compose(Flatten(), ToFloat32())
    arr: np.ndarray = transform(arr)
    assert (
        arr.dtype == np.float32
        and arr.shape == (4,)
        and str(transform) == "Composed Transforms:扁平化->转float32(单精度浮点)"
    )
