r"""一元函数数据集
一元函数数据集数据展示-sin函数
"""

import matplotlib.pyplot as plt
import numpy as np

from mtorch.examples.data._util import create_subplot
from mtorch.utils.data.datasets import UnivariateFunctionDataset

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


def sin(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


dataset = UnivariateFunctionDataset(
    func=sin, x_data=np.arange(0, 1, 0.01)[:, np.newaxis]
)
result = dataset[0 : len(dataset)]
axes = create_subplot()
axes.scatter(result.data, result.label)
plt.tight_layout()
plt.show()
