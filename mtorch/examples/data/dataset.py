from mtorch.utils.data.datasets import SprialDataset, UnivariateFunctionDataset
import numpy as np


def sin(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


sin_dataset = UnivariateFunctionDataset(func=sin)
sprial_datset = SprialDataset()
sin_dataset.graph(title="Sin", imdiately_show=False)
sprial_datset.graph()
