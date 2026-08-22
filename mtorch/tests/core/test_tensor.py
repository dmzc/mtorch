from mtorch import Tensor
import numpy as np


def test_tensor():

    # 测试基本运算符的优先级
    # 加法操作符算子
    result: Tensor = Tensor(1) + 2
    assert isinstance(result, Tensor) and result.data.tolist() == 3, "tensor和数字相加"
    result = 2 + Tensor(1)
    assert isinstance(result, Tensor) and result.data.tolist() == 3, "数字和tensor相加"
    result = Tensor(2) + Tensor(1)
    assert (
        isinstance(result, Tensor) and result.data.tolist() == 3
    ), "tensor和tensor相加"
    result: Tensor = Tensor(1) + np.array(2)
    assert (
        isinstance(result, Tensor) and result.data.tolist() == 3
    ), "tensor和np.ndarray相加"
    result: Tensor = np.array(2) + Tensor(1)
    assert (
        isinstance(result, Tensor) and result.data.tolist() == 3
    ), "np.ndarray和tensor相加"

    # 减法操作符算子
    result: Tensor = Tensor(1) - 2
    assert isinstance(result, Tensor) and result.data.tolist() == -1, "tensor和数字相减"
    result = 2 - Tensor(1)
    assert isinstance(result, Tensor) and result.data.tolist() == 1, "数字和tensor相减"
    result = Tensor(2) - Tensor(1)
    assert (
        isinstance(result, Tensor) and result.data.tolist() == 1
    ), "tensor和tensor相减"
    result: Tensor = Tensor(1) - np.array(2)
    assert (
        isinstance(result, Tensor) and result.data.tolist() == -1
    ), "tensor和np.ndarray相减"
    result: Tensor = np.array(2) - Tensor(1)
    assert (
        isinstance(result, Tensor) and result.data.tolist() == 1
    ), "np.ndarray和tensor相减"

    # 乘法操作符算子
    result: Tensor = Tensor(1) * 2
    assert isinstance(result, Tensor) and result.data.tolist() == 2, "tensor和数字相乘"
    result = 2 * Tensor(1)
    assert isinstance(result, Tensor) and result.data.tolist() == 2, "数字和tensor相乘"
    result = Tensor(2) * Tensor(1)
    assert (
        isinstance(result, Tensor) and result.data.tolist() == 2
    ), "tensor和tensor相乘"
    result: Tensor = Tensor(1) * np.array(2)
    assert (
        isinstance(result, Tensor) and result.data.tolist() == 2
    ), "tensor和np.ndarray相乘"
    result: Tensor = np.array(2) * Tensor(1)
    assert (
        isinstance(result, Tensor) and result.data.tolist() == 2
    ), "np.ndarray和tensor相乘"

    # 乘法操作符算子
    result: Tensor = Tensor(1) / 2
    assert (
        isinstance(result, Tensor) and result.data.tolist() == 0.5
    ), "tensor和数字相除"
    result = 2 / Tensor(1)
    assert isinstance(result, Tensor) and result.data.tolist() == 2, "数字和tensor相除"
    result = Tensor(2) / Tensor(1)
    assert (
        isinstance(result, Tensor) and result.data.tolist() == 2
    ), "tensor和tensor相除"
    result: Tensor = Tensor(1) / np.array(2)
    assert (
        isinstance(result, Tensor) and result.data.tolist() == 0.5
    ), "tensor和np.ndarray相除"
    result: Tensor = np.array(2) / Tensor(1)
    assert (
        isinstance(result, Tensor) and result.data.tolist() == 2
    ), "np.ndarray和tensor相除"

    # 加法逆元
    result: Tensor = -Tensor(1)
    assert isinstance(result, Tensor) and result.data.tolist() == -1, "加法逆元"

    # 幂乘操作
    result: Tensor = Tensor(2) ** 2
    assert isinstance(result, Tensor) and result.data.tolist() == 4, "幂乘操作"

    # 测试原地复合赋值，因为Tensor没有实现iadd、isub之类
    tensor1: Tensor = Tensor(2)
    tensor2 = tensor1
    tensor1 += 2
    assert (
        tensor2 != tensor1 and tensor1.data.tolist() == 4
    ), "加号原地复合赋值会产出新节点"

    tensor1: Tensor = Tensor(2)
    tensor2 = tensor1
    tensor1 -= 2
    assert (
        tensor2 != tensor1 and tensor1.data.tolist() == 0
    ), "减号原地复合赋值会产出新节点"

    tensor1 = Tensor(2)
    tensor2 = tensor1
    tensor1 /= 2
    assert (
        tensor2 != tensor1 and tensor1.data.tolist() == 1
    ), "除号原地复合赋值会产出新节点"

    tensor1 = Tensor(2)
    tensor2 = tensor1
    tensor1 *= 2
    assert (
        tensor2 != tensor1 and tensor1.data.tolist() == 4
    ), "乘号原地复合赋值会产出新节点"
