r"""螺旋数据集
螺旋数据集数据展示
"""

import matplotlib.pyplot as plt
import numpy as np

from mtorch.examples.data._util import create_subplot
from mtorch.utils.data.datasets import Sprial

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
plt.rcParams["axis.unicode_minus"] = False  # 正常显示负号


dataset = Sprial()
axis = create_subplot(title="螺旋数据集")
markers, colors = dataset.get_category_info()
result = dataset[0 : len(dataset)]
data = result.data
label = result.label
for idx in range(len(data)):
    category = label[idx]
    item: np.ndarray = data[idx]
    axis.scatter(
        item[0],
        item[1],
        s=40,
        marker=markers[category],
        c=colors[category],
    )
plt.tight_layout()
plt.show()
