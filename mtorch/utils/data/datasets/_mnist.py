from pathlib import Path

from mtorch._interfaces import DatasetData, Dataset_Type, ITransform
from ._dataset import AbstractDataset
from ._idx import IDXDataset
from typing import Optional


class Mnist(AbstractDataset):

    _root_dir: Path

    _dataset_type: Dataset_Type

    _ensured: bool

    _data_idx_dataset: Optional[IDXDataset]

    _label_idx_dataset: Optional[IDXDataset]

    def __init__(
        self,
        root_dir: str | Path,
        dataset_type: Dataset_Type,
        data_transform: ITransform = None,
        label_transform: ITransform = None,
    ):
        super().__init__(data_transform=data_transform, label_transform=label_transform)
        root_dir = Path(root_dir)
        if root_dir.exists() and not root_dir.is_dir():
            raise NotADirectoryError("root must point to an existing directory.")
        self._root_dir = root_dir
        self._dataset_type = dataset_type
        self._ensured = False
        self._data_idx_dataset = None
        self._label_idx_dataset = None

    def len(self):
        self._ensure()
        return len(self._data_idx_dataset)

    def get_item(self, slices: int | list[int]) -> DatasetData:
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
