from pathlib import Path

# TODO:这应该只在examples中显示
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage

from mtorch._interfaces import DatasetData, IDataset, _Dataset_Type
from mtorch.utils.data.datasets.idx import IDXDataset


class Mnist(IDataset):

    _root_dir: Path

    _dataset_type: _Dataset_Type

    _ensured: bool

    _data_idx_dataset: IDXDataset

    _label_idx_dataset: IDXDataset

    def __init__(self, root_dir: str | Path, dataset_type: _Dataset_Type):
        super().__init__()
        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            raise NotADirectoryError("root must point to an existing directory.")
        self._root_dir = root_dir
        self._dataset_type = dataset_type
        self._ensured = False
        self._data_idx_dataset = None
        self._label_idx_dataset = None

    def __len__(self):
        self._ensure()
        return len(self._data_idx_dataset)

    def __getitem__(self, slices: int | list[int]) -> DatasetData:
        self._ensure()
        return DatasetData(
            data=self._data_idx_dataset[slices], label=self._label_idx_dataset[slices]
        )

    def _ensure(self) -> None:
        if self._ensured:
            if self._data_idx_dataset is None:
                raise ValueError("The Mnist can not correctly initialized!")
            return
        self._ensured = True
        url_base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
        key_file = {
            "train_img": "train-images-idx3-ubyte.gz",
            "train_label": "train-labels-idx1-ubyte.gz",
            "test_img": "t10k-images-idx3-ubyte.gz",
            "test_label": "t10k-labels-idx1-ubyte.gz",
        }
        root_dir: Path = self._root_dir

        data_name = key_file[f"{self._dataset_type}_img"]
        label_name = key_file[f"{self._dataset_type}_label"]
        data_file = root_dir / data_name
        label_file = root_dir / label_name

        def _ensure_root_dir():
            if not root_dir.exists():
                root_dir.mkdir(parents=True, exist_ok=True)

        if not data_file.exists():

            _ensure_root_dir()
            import urllib

            urllib.request.urlretrieve(f"{url_base}{data_name}", str(data_file))
        if not label_file.exists():
            _ensure_root_dir()
            import urllib

            urllib.request.urlretrieve(f"{url_base}{label_name}", str(label_file))

        self._data_idx_dataset = IDXDataset(
            file=data_file, work_dir=root_dir, all_data=True
        )
        self._label_idx_dataset = IDXDataset(file=label_file, work_dir=root_dir)

    def graph(self):

        index = -1
        count = len(self)

        axes_image: AxesImage = None

        def _draw_image(
            is_next=True,
        ) -> tuple[np.ndarray, int] | None:
            nonlocal index, axes_image
            t_idx = None
            if is_next:
                t_idx = index + 1
            else:
                t_idx = index - 1
            if t_idx < 0 or t_idx > count - 1:
                return
            index = t_idx
            result = self[t_idx]
            data = result.data[0]
            label = result.label[0]
            if axes_image is None:
                axes_image = ax.imshow(data, cmap="gray")
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

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")

        _draw_image()

        fig.canvas.mpl_connect("key_press_event", _on_key)
        plt.tight_layout()
        plt.show(block=True)
