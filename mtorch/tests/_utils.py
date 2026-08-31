from contextlib import contextmanager
import importlib.util
import sys

from mtorch import CACHE_DIR
import numpy as np
from mtorch._interfaces import DataArray, ITensor
from mtorch import Tensor
from typing import Callable, Any, Literal
from mtorch.autograd import numerical_diff
import mtorch.core.operator as F
from mtorch.core._core import _softmax, _logsoftmax


TEST_DIRS = CACHE_DIR


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


class DiffUtils:

    @staticmethod
    def _get_aray_func(
        operator: Literal["+", "-", "*", "/"],
    ) -> Callable[[Any, Any], Any]:
        def _func(v1: Any, v2: Any):
            match operator:
                case "+":
                    return v1 + v2
                case "-":
                    return v1 - v2
                case "*":
                    return v1 * v2
                case "/":
                    return v1 / v2

        return _func

    @staticmethod
    def numerical_basic(
        proj_v: DataArray, inputs: list[Any], operator: Literal["+", "-", "*", "/"]
    ) -> list[DataArray]:
        temp: list[DataArray] = []
        for item in inputs:
            temp.append(np.array(item))
        inputs = temp

        return numerical_diff(
            func=DiffUtils._get_aray_func(operator=operator),
            inputs=inputs,
            proj_v=proj_v,
        )

    @staticmethod
    def backward_basic(
        proj_v: Any, inputs: list[Any], operator: Literal["+", "-", "*", "/"]
    ) -> list[DataArray]:
        v1 = Tensor(data=inputs[0])
        v2 = Tensor(data=inputs[1])
        v3: ITensor = DiffUtils._get_aray_func(operator=operator)(v1, v2)
        v3.grad = np.array(proj_v)
        v3.backward()
        return [v1.grad, v2.grad]

    @staticmethod
    def numerical_add(proj_v: DataArray, inputs: list[DataArray]) -> list[DataArray]:
        return DiffUtils.numerical_basic(proj_v=proj_v, inputs=inputs, operator="+")

    @staticmethod
    def backward_add(proj_v: DataArray, inputs: list[DataArray]) -> list[DataArray]:
        return DiffUtils.backward_basic(proj_v=proj_v, inputs=inputs, operator="+")

    @staticmethod
    def numerical_sub(proj_v: DataArray, inputs: list[DataArray]) -> list[DataArray]:
        return DiffUtils.numerical_basic(proj_v=proj_v, inputs=inputs, operator="-")

    @staticmethod
    def backward_sub(proj_v: DataArray, inputs: list[DataArray]) -> list[DataArray]:
        return DiffUtils.backward_basic(proj_v=proj_v, inputs=inputs, operator="-")

    @staticmethod
    def numerical_mul(proj_v: DataArray, inputs: list[DataArray]) -> list[DataArray]:
        return DiffUtils.numerical_basic(proj_v=proj_v, inputs=inputs, operator="*")

    @staticmethod
    def backward_mul(proj_v: DataArray, inputs: list[DataArray]) -> list[DataArray]:
        return DiffUtils.backward_basic(proj_v=proj_v, inputs=inputs, operator="*")

    @staticmethod
    def numerical_div(proj_v: DataArray, inputs: list[DataArray]) -> list[DataArray]:
        return DiffUtils.numerical_basic(proj_v=proj_v, inputs=inputs, operator="/")

    @staticmethod
    def backward_div(proj_v: DataArray, inputs: list[DataArray]) -> list[DataArray]:
        return DiffUtils.backward_basic(proj_v=proj_v, inputs=inputs, operator="/")

    def backward_softmax(proj_v: Any, inputs: list[Any]) -> DataArray:
        input: ITensor = Tensor(inputs[0])
        output: ITensor = F.softmax(input)
        output.grad = np.array(proj_v)
        output.backward()
        return [input.grad]

    def numerical_softmax(proj_v: Any, inputs: list[Any]) -> DataArray:
        def _func(input: DataArray):
            return _softmax(input, axis=1)

        temp: list[DataArray] = []
        for item in inputs:
            temp.append(np.array(item))
        inputs: list[DataArray] = temp
        return numerical_diff(
            func=_func,
            inputs=inputs,
            proj_v=proj_v,
        )

    def backward_logSoftmax(proj_v: Any, inputs: list[Any]) -> DataArray:
        input: ITensor = Tensor(inputs[0])
        output: ITensor = F.logSoftmax(input)
        output.grad = np.array(proj_v)
        output.backward()
        return [input.grad]

    def numerical_logSoftmax(proj_v: Any, inputs: list[Any]) -> DataArray:
        def _func(input: DataArray):
            return _logsoftmax(input, axis=1)

        temp: list[DataArray] = []
        for item in inputs:
            temp.append(np.array(item))
        inputs: list[DataArray] = temp
        return numerical_diff(
            func=_func,
            inputs=inputs,
            proj_v=proj_v,
        )

    def backward_crossEntroyLoss(proj_v: Any, inputs: list[Any]) -> DataArray:
        x: ITensor = Tensor(inputs[0])
        output: ITensor = F.crossEntropyLoss(x, inputs[1])
        output.grad = np.array(proj_v)
        output.backward()
        return [x.grad]

    def numerical_crossEntroyLoss(proj_v: Any, inputs: list[Any]) -> DataArray:
        def _func(x: DataArray, t: DataArray):
            return F.CrossEntropyLoss().forward(x, t)

        inputs: list[DataArray] = [
            np.astype(np.array(inputs[0]), np.float64),
            np.array(inputs[1]),
        ]
        return numerical_diff(
            func=_func,
            inputs=inputs,
            params=[inputs[0]],
            proj_v=proj_v,
        )

    @staticmethod
    def assert_consistency(
        inputs: list[DataArray],
        backward_func: Callable[[DataArray, list[DataArray]], list[DataArray]],
        numerical_func: Callable[[DataArray, list[DataArray]], list[DataArray]],
        msg: str,
        output_shape: tuple[int],
        sample_size: int = 10,
    ):
        flag = True
        for _ in range(sample_size):
            proj_v = np.random.rand(*output_shape)
            backward_grads = backward_func(proj_v, inputs)
            numerical_grads = numerical_func(proj_v, inputs)
            grad_count = len(backward_grads)
            if grad_count != len(numerical_grads):
                flag = False
                break
            for idx in range(grad_count):
                if not np.allclose(numerical_grads[idx], backward_grads[idx]):
                    flag = False
                    break
        assert flag, msg
