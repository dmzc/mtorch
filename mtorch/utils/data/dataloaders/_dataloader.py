from mtorch._interfaces import IDataLoader, IDataset
import math
import numpy as np
from collections.abc import Callable


class DataLoader(IDataLoader):

    _current_iteration: int
    _max_iteration: int
    __initialized: bool

    _batch_size: int
    _shuffle: bool
    _drop_last: bool
    _indexes: list[int]
    _complete_strategy: Callable[[list[int], int, int], list[int]]
    _init_max_iteration: int

    def __init__(
        self,
        dataset: IDataset,
        batch_size: int = None,
        shuffle: bool = True,
        drop_last: bool = False,
        complete_strategy: Callable[[np.ndarray, int, int], np.ndarray] = None,
        max_iteration: int = None,
    ):
        r"""
        Args:
            dataset: 数据集实例。
            batch_size: 每个批次的样本数量，如果不传递，batch_size等于样本长度。
            shuffle: 是否对样本索引进行随机打乱。
            drop_last: 当最后一批样本数量不足 batch_size 时，是否直接丢弃该批次。此选项为True时，`complete_strategy`不生效。
            complete_strategy:
                最后一批样本不足 batch_size 时的补齐回调策略。
                函数签名：(indexes: np.ndarray, complete_size: int, batch_size: int) -> np.ndarray
                indexes: 剩余不足批次的样本索引数组
                complete_size: 需要补全的个数
                batch_size: 批次大小
                返回补充的索引。
            max_iteration: 最大迭代轮数
        """
        super().__init__()
        self._dataset = dataset
        if batch_size is None:
            self._batch_size = len(dataset)
        else:
            self._batch_size = batch_size

        self._shuffle = shuffle
        self._drop_last = drop_last
        self._complete_strategy = complete_strategy
        self._init_max_iteration = max_iteration
        self._max_iteration = None
        self._current_iteration = None
        self._indexes = None
        self.__initialized = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.__initialized:  # 初始化
            self.__do_init()
        if self._max_iteration <= self._current_iteration:  # 样本都迭代完毕
            self.reset()
            raise StopIteration()
        start_index = self._current_iteration * self._batch_size
        end_index = (self._current_iteration + 1) * self._batch_size
        self._current_iteration += 1
        dataset_indexes = self._indexes
        indexes: list[int] = dataset_indexes[start_index:end_index]
        if self._current_iteration == self._max_iteration:  # 最后一轮
            if self._complete_strategy is None or end_index + 1 == len(
                dataset_indexes
            ):  # 没有补全策略，或者最后数据不需要补全
                return self._dataset[indexes]
            complete_indexes = self._complete_strategy(
                dataset_indexes,
                len(dataset_indexes) - 1 - start_index,
                self._batch_size,
            )
            indexes.extend(complete_indexes)
        return self._dataset[indexes]

    def __do_init(self) -> None:
        self.__initialized = True
        dataset_size = len(self._dataset)
        infer_max_iteration: int = None
        if self._drop_last:
            infer_max_iteration = math.floor(dataset_size / self._batch_size)
        else:
            infer_max_iteration = math.ceil(dataset_size / self._batch_size)
        if self._init_max_iteration is None:
            self._max_iteration = infer_max_iteration
        else:
            self._max_iteration = min(self._init_max_iteration, infer_max_iteration)
        self._current_iteration = 0
        if self._shuffle:
            self._indexes = np.random.permutation(dataset_size).tolist()
        else:
            self._indexes = [idx for idx in range(0, dataset_size)]

    def reset(self) -> None:
        self.__initialized = False
