r"""mnist数据集
mnist数据集数据展示
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage

from mtorch import ROOT_DIR
from mtorch.examples.data._util import create_subplot
from mtorch.utils.data.datasets import Mnist

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


dataset = Mnist(root_dir=ROOT_DIR / "utils/data/resources/MNIST", dataset_type="train")
index = -1
count = len(dataset)
axes_image: AxesImage = None


def _draw_image(
    is_next=True,
) -> tuple[np.ndarray, int] | None:
    global index, axes_image
    t_idx = None
    if is_next:
        t_idx = index + 1
    else:
        t_idx = index - 1
    if t_idx < 0 or t_idx > count - 1:
        return
    index = t_idx
    result = dataset[t_idx]
    data = result.data[0]
    label = result.label[0]
    if axes_image is None:
        axes_image = ax.imshow(data, cmap="gray", interpolation="nearest")
    else:
        axes_image.set_data(data)
    ax.set_title(f"[{t_idx+1}/{count}] label = {label}")
    fig.canvas.draw_idle()


def _on_key(event):
    if event.key == "right":
        _draw_image()
    elif event.key == "left":
        _draw_image(is_next=False)
    elif event.key == "q":
        plt.close(fig)
        return


ax = create_subplot()
fig = ax.figure
ax.axis("off")
_draw_image()
fig.canvas.mpl_connect("key_press_event", _on_key)
plt.tight_layout()
plt.show(block=True)
