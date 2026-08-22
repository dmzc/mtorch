import mtorch.utils as mutils
from mtorch.tests import TEST_DIRS, mock_missing_package
import numpy as np
from pathlib import Path
from collections.abc import Callable

current_filename = Path(__file__).stem


def _clean(path: Path):

    if not path.is_dir() or not path.exists():
        return
    if not str(TEST_DIRS) in str(path):
        return
    if any(path.iterdir()):
        import shutil

        shutil.rmtree(path)


def _default_func(
    max_iteration: int, per_count: int, current_iteration: int
) -> np.ndarray | list[np.ndarray]:
    if per_count == 1:
        return [12, 34, 45]
    else:
        return np.ones((per_count, 3))


class _MockDumpDataIterator:
    r"""
    模拟dump数据迭代器。

    用于生成模拟的批量dump数据，按轮次迭代调用数据获取函数产出数据。

    Args:
        count: 总迭代轮数，迭代一共执行多少轮
        per_count: 每轮产出的数据条数，单次迭代返回的数据量
        func: 数据获取回调函数。
            入参依次为：总迭代轮数、每轮数据条数、当前迭代轮号（从0开始）。
            返回值：返回单条对象时，不允许在套一层数组。
    """

    _max_iteration: int
    _current_iteration: int
    _per_count: int
    _func: Callable[[int, int, int], any]

    _initialized: bool

    def __init__(
        self,
        max_iteration: int = 10,
        per_count: int = 1,
        func: Callable[[int, int, int], any] = _default_func,
    ):
        self._max_iteration = max_iteration
        self._per_count = per_count
        self._func = func
        self._current_iteration = 0
        self._initialized = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._initialized == False:
            self._initialized = True
        if self._current_iteration == self._max_iteration:
            self._initialized = False
            self._current_iteration = 0
            raise StopIteration()
        current_iteration = self._current_iteration
        self._current_iteration += 1
        return self._func(self._max_iteration, self._per_count, current_iteration)

    def tolist(self):
        list1 = []
        is_single = self._per_count == 1
        for items in self:
            if is_single == 1:
                if isinstance(items, np.ndarray):
                    items = items.tolist()
                list1.append(items)
            else:
                for item in items:
                    if isinstance(item, np.ndarray):
                        item = item.tolist()
                    list1.append(item)
        return list1


def test_dumps():
    file_dir = TEST_DIRS / current_filename / "test_dumps"
    filename: Path = file_dir / "test"
    _clean(file_dir)

    data = np.eye(10, 10)

    # 1. 不压缩直接落地
    t_file = mutils.dumps(file=filename, obj=data, compress=False)

    load_data = mutils.loads(file=t_file)

    assert (
        str(t_file).endswith(".json") and load_data == data.tolist()
    ), "数据能正常dumps、loads（不压缩）"

    # 2. 不压缩直接落地（缺少orjson）
    with mock_missing_package("orjson"):

        t_file = mutils.dumps(file=filename, obj=data, compress=False)

        load_data = mutils.loads(file=t_file)

        assert (
            str(t_file).endswith(".json") and load_data == data.tolist()
        ), "没有安装orjson，数据能正常dumps、loads（不压缩）"

    # 3. 压缩直接落地
    t_file = mutils.dumps(file=filename, obj=data, compress=True)

    load_data = mutils.loads(file=t_file)
    assert (
        str(t_file).endswith(".zst") and load_data == data.tolist()
    ), "数据能正常dumps、loads（压缩）"

    # 4. 压缩直接落地(缺少zstandard)
    with mock_missing_package("zstandard"):
        t_file = mutils.dumps(file=filename, obj=data, compress=True)

        load_data = mutils.loads(file=t_file)
        assert (
            str(t_file).endswith(".gzip") and load_data == data.tolist()
        ), "缺少zstandard时，数据能正常dumps、loads（压缩）"

    # 5. 新文件会覆盖就文件
    data1 = np.eye(5, 5)
    data2 = np.eye(10, 10)
    file1 = mutils.dumps(file=filename, obj=data1, compress=False)
    file2 = mutils.dumps(file=filename, obj=data2, compress=False)
    load_data = mutils.loads(file=file1)
    assert (
        str(file1) == str(file2) and load_data == data2.tolist()
    ), "新数据会覆盖旧数据"


def test_dumps_stream():
    file_dir = TEST_DIRS / current_filename / "test_dumps_stream"
    filename: Path = file_dir / "test"
    _clean(file_dir)
    per_count = 2
    data = _MockDumpDataIterator(per_count=per_count, max_iteration=20)

    # 1. 不压缩直接落地
    t_file = mutils.dumps_stream(file=filename, iteratable_obj=data, compress=False)

    load_data = mutils.loads(file=t_file)

    assert (
        str(t_file).endswith(".json") and load_data == data.tolist()
    ), "数据能流式dumps、loads（不压缩）"

    # 2. 不压缩直接落地（缺少orjson）
    with mock_missing_package("orjson"):

        t_file = mutils.dumps_stream(file=filename, iteratable_obj=data, compress=False)

        load_data = mutils.loads(file=t_file)

        assert (
            str(t_file).endswith(".json") and load_data == data.tolist()
        ), "没有安装orjson，数据能正常流式dumps、loads（不压缩）"

    # 3. 压缩直接落地
    t_file = mutils.dumps_stream(file=filename, iteratable_obj=data, compress=True)

    load_data = mutils.loads(file=t_file)
    assert (
        str(t_file).endswith(".zst") and load_data == data.tolist()
    ), "数据能正常dumps、loads（压缩）"

    # 4. 压缩直接落地(缺少zstandard)
    with mock_missing_package("zstandard"):
        t_file = mutils.dumps_stream(file=filename, iteratable_obj=data, compress=True)

        load_data = mutils.loads(file=t_file)
        assert (
            str(t_file).endswith(".gzip") and load_data == data.tolist()
        ), "缺少zstandard时，数据能正常dumps、loads（压缩）"

    # 5. 新文件会覆盖就文件
    data1 = _MockDumpDataIterator(per_count=10, max_iteration=10)
    data2 = _MockDumpDataIterator(per_count=5, max_iteration=5)
    _clean(file_dir)
    file1 = mutils.dumps_stream(file=filename, iteratable_obj=data1, compress=False)
    file2 = mutils.dumps_stream(file=filename, iteratable_obj=data2, compress=False)
    load_data = mutils.loads(file=file1)
    assert (
        str(file1) == str(file2) and load_data == data2.tolist()
    ), "新数据会覆盖旧数据"
