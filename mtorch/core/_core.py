import numpy as np
from typing import Any
from mtorch._interfaces import DataArray


def _eye(N, M=None, k=0, dtype=float) -> DataArray:
    return np.eye(N=N, M=M, k=k, dtype=dtype)


def _zeros_like(x: DataArray) -> DataArray:
    return np.zeros_like(x)


def _argsort(obj: Any):
    return np.argsort(obj)


def _raise_invalid_ndim_error(shape: tuple[int], to_shape: tuple[int]) -> None:
    raise ValueError(
        f"不符合SumTo的形状规则：\n原始形状：{shape}\n目标形状：{to_shape}"
    )


def _log(x: DataArray) -> DataArray:
    return np.log(x)


def _sum(x: DataArray, axis: tuple[int] | int = None, keepdims=False) -> DataArray:
    return np.sum(x, axis=axis, keepdims=keepdims)


def _broadcast_to(x: DataArray, shape: tuple[int]) -> DataArray:
    return np.broadcast_to(x, shape=shape)


def _sin(x: DataArray) -> DataArray:
    return np.sin(x)


def _cos(x: DataArray) -> DataArray:
    return np.cos(x)


def _exp(x: DataArray) -> DataArray:
    return np.exp(x)


def _tanh(x: DataArray) -> DataArray:
    return np.tanh(x)


def _sum_to(x: DataArray, shape: tuple[int]) -> DataArray:
    f_shape = x.shape
    t_shape = shape
    if f_shape == t_shape:
        return x
    f_ndim = len(f_shape)
    t_ndim = len(t_shape)
    diff = f_ndim - t_ndim
    if diff < 0:
        _raise_invalid_ndim_error(shape=f_shape, to_shape=t_shape)

    sum_axis: list[int] = list(range(diff))  # 那些轴要求和，前面多的维度肯定要求和

    for f_index in range(diff, f_ndim):
        t_dim = t_shape[f_index - diff]
        if f_shape[f_index] == t_dim:
            continue
        if t_dim == 1:
            sum_axis.append(f_index)
            continue
        # 目标维既不是1，也不相等，无法处理
        _raise_invalid_ndim_error(shape=f_shape, to_shape=t_shape)
    result: DataArray = np.sum(x, axis=tuple(sum_axis), keepdims=True)
    if diff > 0:  # 前面多的维度被压缩了，要裁剪掉
        result = result.squeeze()
    return result


def _maximum(x: DataArray, num: float) -> DataArray:
    return np.maximum(x, num)


def _max(x: DataArray, axis: tuple[int], keepdims: bool) -> DataArray:
    return np.max(x, axis=axis, keepdims=keepdims)


def _matmul(left: DataArray, right: DataArray) -> DataArray:
    return np.dot(left, right)


def _reshape(x: DataArray, shape: tuple[int]) -> DataArray:
    return np.reshape(x, shape)


def _transpose(x: DataArray, axis: tuple[int] = None) -> DataArray:
    return np.transpose(x, axes=axis)


def _ones_like(x: DataArray) -> DataArray:
    return np.ones_like(x)


def _data_array(x: object) -> DataArray:
    return np.array(x)


def _isscalar(x: object) -> bool:
    return np.isscalar(x)


def _softmax(x: DataArray, axis: int) -> DataArray:
    """
    无损数值优化，数学等价、梯度不变，专门压制exp的浮点溢出/下溢，是工业实现
    Softmax的标准固定写法。
    """
    xmax = _max(x, axis=axis, keepdims=True)
    _max(x, axis=axis, keepdims=True)
    y = _exp(x - xmax)
    y /= _sum(y, axis=axis, keepdims=True)
    return y


def _logsoftmax(x: DataArray, axis: int) -> DataArray:
    xmax = _max(x, axis=axis, keepdims=True)
    y = _exp(x - xmax)
    return x - xmax - _log(_sum(y, axis=axis, keepdims=True))
