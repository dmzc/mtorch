import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def create_subplot(
    figsize: tuple[int] = (6, 4),
    show_toolbar: bool = False,
    title: str | None = None,
) -> Axes:
    if title is None:
        title = " "
    figure, axes = plt.subplots(figsize=figsize)
    figure.canvas.manager.set_window_title(title)
    if not show_toolbar:
        figure.canvas.manager.toolbar.setVisible(False)  # 去掉toolbar显示
    return axes
