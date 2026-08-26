from pathlib import Path
from contextlib import contextmanager
import importlib.util
import sys

from mtorch.utils.data.datasets import AbstractDataset
import numpy as np


TEST_DIRS = Path(__file__).parent / "_dirs"


@contextmanager
def mock_missing_package(pkg_name: str):
    """
    模拟某个包完全不存在，欺骗 find_spec + import。
    零 unittest.mock 依赖，纯标准库。
    拦截主包以及所有子模块 pkg_name.*
    """
    # 1. 保存原始状态
    real_find_spec = importlib.util.find_spec

    r"""
        1. `...` = `Ellipsis` 对象，单例；
        2. 此处作为**哨兵 (sentinel) 标记**，区分：`sys.modules原本没有该key`；
        3. 不能用`None`当哨兵，因为`sys.modules`合法值可以是`None`。
    """
    orig_sys_mod_val = sys.modules.get(pkg_name, ...)

    # 2. 替换 find_spec
    def fake_find(name: str):
        if name == pkg_name or name.startswith(f"{pkg_name}."):
            return None
        return real_find_spec(name)

    r"""
        `try‑finally` 只负责保护 `yield`（用户代码块）；前面的修改全局状态的初始化代码，放在 try 之外。
    """
    importlib.util.find_spec = fake_find
    # 设置 sys.modules 为 None，触发 import 报 ModuleNotFoundError
    sys.modules[pkg_name] = None

    try:
        yield
    finally:
        # 3. 严格复原，一定要回滚全局状态
        importlib.util.find_spec = real_find_spec
        if orig_sys_mod_val is ...:
            sys.modules.pop(pkg_name, None)
        else:
            sys.modules[pkg_name] = orig_sys_mod_val


class MockDataset(AbstractDataset):

    mock_data: np.ndarray
    mock_label: np.ndarray

    def __init__(self, data: np.ndarray = None, label: np.ndarray = None):
        super().__init__()
        if data is not None and not isinstance(data, np.ndarray):
            data = np.array(data)
        if label is not None and not isinstance(label, np.ndarray):
            label = np.array(label)
        self.mock_data = data
        self.mock_label = label

    def load_data(self) -> tuple[np.ndarray] | np.ndarray:

        if self.mock_label is not None:
            return (self.mock_data, self.mock_label)
        else:
            return self.mock_data
