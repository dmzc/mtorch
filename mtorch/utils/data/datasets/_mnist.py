from pathlib import Path

from mtorch import DatasetData, Dataset_Type, ITransform, CACHE_DIR
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
        dataset_type: Dataset_Type,
        root_dir: str | Path = None,
        data_transform: ITransform = None,
        label_transform: ITransform = None,
    ):
        super().__init__(data_transform=data_transform, label_transform=label_transform)
        if root_dir is not None:
            root_dir = Path(root_dir)
            if root_dir.exists() and not root_dir.is_dir():
                raise NotADirectoryError("root must point to an existing directory.")
        else:
            root_dir = CACHE_DIR / "MNIST"
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
            self._download_file(f"{url_base}{data_name}", str(data_file))
        if not label_file.exists():
            _ensure_root_dir()
            self._download_file(f"{url_base}{label_name}", str(label_file))

        self._data_idx_dataset = IDXDataset(
            file=data_file, work_dir=root_dir, all_data=True
        )
        self._label_idx_dataset = IDXDataset(file=label_file, work_dir=root_dir)

    def _download_file(self, url: str, file: str):
        def _download_progress(block_num, block_size, total_size):
            import sys

            downloaded = block_num * block_size
            if total_size <= 0:
                # 服务器没返回总大小，只打印已下载字节
                sys.stdout.write(
                    f"\rDownloading: {downloaded / 1024:.1f} KB (unknown total)"
                )
            else:
                percent = min(100.0, downloaded * 100.0 / total_size)
                bar_len = 40
                filled = int(bar_len * percent / 100)
                bar = "█" * filled + "-" * (bar_len - filled)
                sys.stdout.write(
                    f"\r[{bar}] {percent:5.1f}%  {downloaded/1024:.1f}/{total_size/1024:.1f} KB"
                )
            sys.stdout.flush()

        import urllib

        print(f"Start download {url}")
        urllib.request.urlretrieve(url, file, _download_progress)
        print(f"Successfuly download {url}")
