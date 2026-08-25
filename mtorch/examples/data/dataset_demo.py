import numpy as np

from mtorch import ROOT_DIR
from mtorch.utils.data.datasets import Mnist, Sprial, UnivariateFunctionDataset


def sin(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


sin_dataset = UnivariateFunctionDataset(func=sin)
sprial_datset = Sprial()
sin_dataset.graph(title="Sin", imdiately_show=False)
sprial_datset.graph()
mnist = Mnist(root_dir=ROOT_DIR / "utils/data/resources/MNIST", dataset_type="train")
mnist.graph()
