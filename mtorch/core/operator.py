from __future__ import annotations
from abc import abstractmethod
import numpy as np
import weakref
from mtorch._interfaces import ITensor, IOperator, Slice_Type, DataArray
from mtorch.core._tensor import Tensor
from ._core import (
    _sum,
    _sum_to,
    _broadcast_to,
    _matmul,
    _maximum,
    _transpose,
    _reshape,
    _argsort,
    _zeros_like,
    _eye,
    _log,
    _exp,
    _sin,
    _cos,
    _tanh,
    _logsoftmax,
    _softmax,
)
from typing import Any

ENABLE_BACKPROGATION = True


# ==========================================================================
# 算子基类
# ==========================================================================
class Operator(IOperator):

    def __init__(self):
        self.label = self.__class__.__name__
        self.inputs = None
        self.outputs = None
        self.generation = None

    def __call__(self, *xs: tuple[Any]) -> list[ITensor] | ITensor:
        inputs = []
        for x in xs:
            if isinstance(x, Tensor):
                inputs.append(x)
            else:
                inputs.append(Tensor(x))
        xs_data = [x.data for x in inputs]
        ys = self.forward(*xs_data)
        if not isinstance(ys, tuple):
            ys = (ys,)
        creator = None
        if ENABLE_BACKPROGATION:
            self.generation = max([x.generation for x in inputs])
            creator = self
        outputs = [Tensor(data=y, creator=creator) for y in ys]
        if ENABLE_BACKPROGATION:
            self.outputs = [weakref.ref(output) for output in outputs]
            self.inputs = inputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: Any) -> Any:
        raise NotImplementedError

    def backward(self, dout: DataArray) -> Any:
        raise NotImplementedError

    @property
    def id(self) -> str:
        return f"_{id(self)}_"

    @property
    def name(self) -> int:
        ret_name = f"{self.__class__.__name__}"
        if ENABLE_BACKPROGATION:
            ret_name = f"{ret_name}({self.generation})"

        return ret_name


class UrayOperator(Operator):
    """
    二元逐元素算子基类。
    底层基于 DataArray，前向计算遵循 NumPy 广播规则：
    1. 形状从尾部（最右侧维度）对齐，维度数量更少的张量在左侧补长度为1的维度；
    2. 维度补齐后，对应位置的维度尺寸必须相等，或其中一方尺寸为1；
    不满足条件则无法广播，触发形状异常。
    """

    __x1_shape: tuple[int]
    __x2_shape: tuple[int]

    def __init__(self):
        super().__init__()
        self.__x1_shape = None
        self.__x2_shape = None

    def forward(self, x1: DataArray, x2: DataArray):
        self.__x1_shape = x1.shape
        self.__x2_shape = x2.shape

    def backward(self, dout: DataArray) -> list[DataArray]:
        dx1, dx2 = self.get_gradient(dout=dout)
        x1_shape = self.__x1_shape
        x2_shape = self.__x2_shape
        if x1_shape != x2_shape:
            return _sum_to(dx1, x1_shape), _sum_to(dx2, x2_shape)
        return dx1, dx2

    @abstractmethod
    def get_gradient(self, dout: DataArray) -> tuple[DataArray]: ...


# ==========================================================================
# 算子基类
# ==========================================================================


# ==========================================================================
# 基础代数算子
# ==========================================================================


class Add(UrayOperator):
    """
    加法
    """

    def __init__(self):
        super().__init__()

    def forward(self, x1: DataArray, x2: DataArray) -> DataArray:
        super().forward(x1, x2)
        return x1 + x2

    def get_gradient(self, dout: DataArray) -> tuple[DataArray, DataArray]:
        return dout, dout


def add(x1: Any, x2: Any) -> ITensor:
    """
    x1、x2 - IVariable、DataArray、数字 反正都会包装成IVariable
    """
    return Add()(x1, x2)


class Neg(Operator):
    """
    加法逆元
    """

    def forward(self, x: DataArray) -> DataArray:
        return -x

    def backward(self, dout: DataArray) -> DataArray:
        return -dout


def neg(x: Any) -> ITensor:
    return Neg()(x)


class Mul(UrayOperator):
    """
    乘法
    """

    def forward(self, x1: DataArray, x2: DataArray) -> DataArray:
        super().forward(x1=x1, x2=x2)
        return x1 * x2

    def get_gradient(self, dout: DataArray) -> tuple[DataArray]:
        x0, x1 = self.inputs[0], self.inputs[1]
        return x1.data * dout, x0.data * dout


def mul(x1: Any, x2: Any):
    return Mul()(x1, x2)


class Sub(UrayOperator):
    def forward(self, x0: DataArray, x1: DataArray) -> DataArray:
        super().forward(x1=x0, x2=x1)
        y = x0 - x1
        return y

    def get_gradient(self, dout: DataArray) -> list[DataArray]:
        return dout, -dout


def sub(x0: Any, x1: Any):

    # 减法、除法还是在拆解为加法、乘法逆元
    #
    # return x0-Neg(x1)
    return Sub()(x0, x1)


def rsub(x0: Any, x1: Any):
    return Sub()(x1, x0)


class Div(UrayOperator):
    """
    除法
    """

    def forward(self, x0: DataArray, x1: DataArray):
        super().forward(x1=x0, x2=x1)
        return x0 / x1

    def get_gradient(self, dout: DataArray) -> tuple[DataArray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = dout / x1
        gx1 = dout * (-x0 / x1**2)
        return gx0, gx1


def div(x0: Any, x1: Any):
    return Div()(x0, x1)


def rdiv(x0: Any, x1: Any):
    return Div()(x1, x0)


class Pow(Operator):
    """
    乘幂
    """

    def __init__(self, c: int):
        self.c = c

    def forward(self, x: DataArray) -> DataArray:
        y = x**self.c
        return y

    def backward(self, dout: DataArray) -> DataArray:
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * dout
        return gx


def pow(x: Any, c: int):
    return Pow(c)(x)


# ==========================================================================
# 基本代数算子
# ==========================================================================


# ==========================================================================
# 基本超越算子
# ==========================================================================
class Sin(Operator):
    """
    正弦
    """

    def forward(self, x: DataArray) -> DataArray:
        return _sin(x)

    def backward(self, dout: DataArray) -> DataArray:
        return dout * _cos(self.inputs[0].data)


def sin(x: Any) -> ITensor:
    return Sin()(x)


class Cos(Operator):
    """
    余弦
    """

    def forward(self, x: DataArray) -> DataArray:
        return _cos(x)

    def backward(self, dout: DataArray) -> DataArray:
        return dout * -_sin(self.inputs[0].data)


def cos(x) -> ITensor:
    return Cos()(x)


class Tanh(Operator):
    """
    双曲正切
    """

    def forward(self, x: DataArray) -> DataArray:
        return _tanh(x)

    def backward(self, dout: DataArray) -> DataArray:
        y = self.outputs[0]().data  # weakref
        return dout * (1 - y * y)


def tanh(x: Any) -> ITensor:
    return Tanh()(x)


class Exp(Operator):
    """
    指数函数
    """

    def forward(self, x: DataArray) -> DataArray:
        y = _exp(x)
        return y

    def backward(self, dout: DataArray) -> DataArray:
        y = self.outputs[0]().data  # weakref
        return dout * y


def exp(x):
    return Exp()(x)


class Log(Operator):
    """
    对数函数
    """

    def forward(self, x: DataArray):
        y = _log(x)
        return y

    def backward(self, dout: DataArray) -> DataArray:
        x = self.inputs[0].data
        gx = dout / x
        return gx


def log(x):
    return Log()(x)


# ==========================================================================
# 基本超越算子
# ==========================================================================


# ==========================================================================
# 张量形状算子
# ==========================================================================


class Reshape(Operator):
    """
    张量形状变更
    """

    __n_shape: tuple[int]  # 新形状
    __o_shape: tuple[int]  # 旧形状

    def __init__(self, n_shape: tuple[int]):
        super().__init__()
        self.__n_shape = n_shape
        self.__o_shape = None

    def forward(self, x: DataArray) -> DataArray:
        self.__o_shape = x.shape
        return _reshape(x, self.__n_shape)

    def backward(self, dout: DataArray) -> DataArray:
        return _reshape(dout, self.__o_shape)


def reshape(x: DataArray | ITensor | list[int], shape: tuple[int]) -> ITensor:
    """
    list[int]代表原生多维数组
    """
    return Reshape(shape)(x)


class Transpose(Operator):
    """
    transpose:数据本身在内存中不会变,只是改变shape、stride。

    如果不传递参数，那么就是全部倒序下：

    x.shape # (2, 3, 4)

    x = x.transpose()

    x.shape # (4, 3, 2)
    """

    __n_axis: tuple[int]

    def __init__(self, n_axis=None):
        super().__init__()
        self.__n_axis = n_axis

    def forward(self, x: DataArray):
        return _transpose(x, self.__n_axis)

    def backward(self, dout: DataArray) -> DataArray:
        n_axis = self.__n_axis
        if n_axis is None:
            return _transpose(dout)
        axis_len = len(n_axis)
        # TODO:这里没搞懂，要连通np.transpose的算法一起搞清楚
        inv_axis = tuple(_argsort([ax % axis_len for ax in n_axis]))
        return _transpose(dout, inv_axis)


def transpose(x: DataArray | ITensor | list[int], axis=None):
    """
    list[int]代表原生多维数组
    """
    return Transpose(axis)(x)


class GetItem(Operator):

    __slices: Slice_Type

    def __init__(self, slices: Slice_Type):
        super().__init__()
        self.__slices = slices

    def forward(self, x: DataArray) -> DataArray:
        return x[self.__slices]

    def backward(self, dout: DataArray) -> DataArray:

        x_left = _zeros_like(self.inputs[0].data)
        if (
            dout.ndim == 1 and dout.shape[0] == 1
        ):  # 形如 x_left[0,1]形式，左侧为标量，此时右侧如果为np.array会警告
            dout = dout.item()
        x_left[self.__slices] = dout
        return x_left


def get_item(x: Any, slices: Slice_Type) -> ITensor:
    return GetItem(slices)(x)


# ==========================================================================
# 张量形状算子
# ==========================================================================

# ==========================================================================
# 常用张量算子
# ==========================================================================


class Sum(Operator):
    __keepdims: bool
    __axis: tuple[int] | int
    __from_shape: tuple[int]

    def __init__(self, keep_dims: bool = False, axis: int | tuple[int] | None = None):
        super().__init__()
        self.__keepdims = keep_dims
        if axis is not None:
            if isinstance(axis, int):
                self.__axis = (axis,)
            else:
                self.__axis = tuple(axis)
        else:
            self.__axis = None

    def forward(self, x: DataArray) -> int | float:
        self.__from_shape = x.shape
        return _sum(x, axis=self.__axis, keepdims=self.__keepdims)

    def backward(self, dout: DataArray) -> DataArray:
        keepdims = self.__keepdims
        axis = self.__axis
        from_shape = self.__from_shape
        if keepdims:  # 正向传播时求和维度的没有被删除，直接reshape即可
            return _broadcast_to(dout, from_shape)
        if axis is None:  # 所有轴求和，dout此时是一个标量，直接广播即可
            return _broadcast_to(dout, from_shape)
        to_shape: list[int] = list(dout.shape)
        # 补全被删去的求和维度
        # 比如：原来（2，3，4，5，6)，按轴（1，3）求和得到（2，4，6）
        # 还原时需要先还原小索引，这样才不至于破坏后面的所有
        for axis in sorted(axis):
            to_shape.insert(axis, 1)
        return _broadcast_to(_reshape(dout, tuple(to_shape)), from_shape)


def sum(x: Any, axis: tuple[int] | int = None, keepdims=False) -> ITensor:

    return Sum(axis=axis, keep_dims=keepdims)(x)


class BroadcastTo(Operator):
    """
    广播扩展算子
    from_shape - 扩展前张量形状
    to_shape   - 目标扩展形状

    两条约束（遵循标准张量右对齐广播规则）：
    1. len(from_shape) <= len(to_shape)；
       若from_shape维度更少，则在左侧（前置）自动补长度为1的维度；
    2. 两个形状执行右对齐；对齐后的每一维，尺寸必须相等，或from_shape对应维度尺寸为1。
    """

    __from_shape: tuple[int]
    __to_shape: tuple[int]

    def __init__(self, shape: tuple[int]):
        super().__init__()
        self.__to_shape = shape
        self.__from_shape = None

    def forward(self, x: DataArray) -> DataArray:
        from_shape = self.__from_shape = x.shape
        if from_shape == self.__to_shape:
            return x
        return _broadcast_to(x, self.__to_shape)

    def backward(self, dout: DataArray) -> DataArray:
        return _sum_to(dout, self.__from_shape)


def broadcast_to(x: DataArray, shape: tuple[int]) -> ITensor:
    return BroadcastTo(shape=shape)(x)


class SumTo(Operator):
    """
    规约对齐求和算子

    shape - 求和前的维度

    to_shape - 求和后的维度

    有一下两点约束：

    1. len(shape) >= len(to_shape)，对于前置多的维度，会被压缩掉
    2. 形状按右对齐，每个维度要想等或to_shape维度为1

    """

    __from_shape: tuple[int]  # 原始形状

    __to_shape: tuple[int]  # 要规约到的形状

    def __init__(self, shape: tuple[int]):
        super().__init__()
        self.__to_shape = shape
        self.__from_shape = None

    def forward(self, x: DataArray) -> DataArray:
        self.__from_shape = x.shape
        return _sum_to(x, shape=self.__to_shape)

    def backward(self, dout: DataArray) -> DataArray:
        return _broadcast_to(dout, self.__from_shape)


def sum_to(x: DataArray, shape: tuple[int]) -> ITensor:
    return SumTo(shape=shape)(x)


class Matmul(Operator):

    def forward(self, x: DataArray, w: DataArray):
        return _matmul(left=x, right=w)

    def backward(self, dout: DataArray) -> DataArray:
        x_data = self.inputs[0].data
        w_data = self.inputs[1].data
        return _matmul(dout, w_data.T), _matmul(x_data.T, dout)


def matmul(x: DataArray, w: DataArray) -> ITensor:
    return Matmul()(x, w)


def rmatmul(w: DataArray, x: DataArray) -> ITensor:
    return Matmul()(x, w)


def imatmul(x: DataArray, w: DataArray) -> ITensor:
    return Matmul()(x, w)


# ==========================================================================
# 常用张量算子
# ==========================================================================


# ==========================================================================
# 损失算子
# ==========================================================================


class MeanSquareLoss(Operator):
    def forward(self, y_actual: DataArray, y_expect: DataArray) -> DataArray:
        diff: DataArray = y_actual - y_expect
        return _sum((diff) ** 2) / diff.shape[0]

    def backward(self, dout: DataArray) -> DataArray:
        y_acutal = self.inputs[0].data
        y_expect = self.inputs[1].data
        diff: DataArray = y_acutal - y_expect
        dy_actual = dout * 2 * diff / diff.shape[0]
        return dy_actual, -dy_actual


def mean_square_loss(y_actual, y_expect) -> ITensor:
    return MeanSquareLoss()(y_actual, y_expect)


class CrossEntropyLoss(Operator):
    """
    softmax + 交叉熵损失
    """

    _axis: int

    def __init__(self, axis: int = 1):
        super().__init__()
        self._axis = axis

    def forward(self, x: DataArray, t: DataArray):
        y = _logsoftmax(x, axis=self._axis)
        N = x.shape[0]
        log_p = y[np.arange(N), t.ravel()]
        return -_sum(log_p) / N

    def backward(self, dout: DataArray) -> DataArray:
        # TODO:这里还需要改善，目前只支持二维数组
        x, t = self.inputs
        N, CLS_NUM = x.shape
        dout *= 1 / N
        y = _softmax(x.data, axis=self._axis)
        # convert to one-hot
        t_onehot = _eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * dout
        return y


def crossEntropyLoss(x, t, axis: int = 1) -> ITensor:
    return CrossEntropyLoss(axis=axis)(x, t)


# ==========================================================================
# 损失算子
# ==========================================================================


# ==========================================================================
# 激活算子
# ==========================================================================


class Sigmoid(Operator):
    def forward(self, x: DataArray) -> DataArray:
        # 传统形式：y=1/(1+np.exp(-x)),做变量代换，令z=x*0.5，两者是等价的。
        # tanh 在 numpy 底层做了稳定算法，不会出现指数直接爆炸计算，大幅度缓解极
        # 端输入下的溢出问题。
        return _tanh(x * 0.5) * 0.5 + 0.5

    def backward(self, dout: DataArray) -> DataArray:
        x = self.outputs[0]().data
        return dout * x * (1 - x)


def sigmoid(x) -> ITensor:
    return Sigmoid()(x)


class ReLU(Operator):
    def forward(self, x: DataArray) -> DataArray:
        return _maximum(x, 0.0)

    def backward(self, dout: DataArray) -> DataArray:
        x = self.inputs[0]
        return dout * (x.data > 0)


def relu(x) -> ITensor:
    return ReLU()(x)


class Softmax(Operator):
    """
    通常只在输出层使用的激活函数
    """

    _axis: int  # 按哪个维度进行softmax，默认最后一个维度

    def __init__(self, axis: int = 1):
        super().__init__()
        self._axis = axis

    def forward(self, x: DataArray) -> DataArray:
        return _softmax(x, axis=self._axis)

    def backward(self, dout: DataArray) -> DataArray:
        ei_div_esum: DataArray = self.outputs[0]().data
        """
        xi - 输入元素
        ei - 分子元素，ei = exp(xi)
        S - 分母元素，S = e1 + e2 + ... + ei
        y - 前向传播输出，y = ei * 1/S
        
        反向传播时：
        
        分子链路：1/S * ei * douti（自然指数求导），即为y
        单个xi的分母链路：ei * sum(ei * douti) * -1/(S平方)，但是这个分母会传播到每个x，
        所以最终xi的导数是所有的相加就约掉了一个x
        
        """
        dxi: DataArray = ei_div_esum * dout  # 单个元素反向传播
        dxsum: DataArray = -ei_div_esum * _sum(
            dxi, axis=self._axis, keepdims=True
        )  # 下方和
        return dxi + dxsum


def softmax(x, axis: int = 1) -> ITensor:
    return Softmax(axis=axis)(x)


class LogSoftmax(Softmax):
    """
    对传统softmax套了一个log，利用对数消除分子的指数的计算，极大减少指数计算的溢
    出。也可以很好的配合交叉熵损失使用。x漂移根据等式来看是恒等变换，所以计算导
    数时不需要考虑偏移
    """

    def forward(self, x):
        return _logsoftmax(x, axis=self._axis)

    def backward(self, dout: DataArray) -> DataArray:
        y: DataArray = self.outputs[0]().data
        return dout - _exp(y) * _sum(dout, axis=self._axis, keepdims=True)


def logSoftmax(x, axis: int = 1) -> ITensor:
    return LogSoftmax(axis=axis)(x)


# ==========================================================================
# 激活算子
# ==========================================================================

# ==========================================================================
# 神经网络常用算子
# ==========================================================================


class Linear(Operator):
    """
    线性仿射算子
    """

    def forward(self, x: DataArray, w: DataArray, b: DataArray):
        out = _matmul(x, w)
        if b is not None:
            out = out + b
        return out

    def backward(self, dout: DataArray) -> tuple[DataArray]:
        x, w, b = self.inputs
        x = self.inputs[0]
        w = self.inputs[1]
        b = self.inputs[2]
        db = None
        if b is not None:
            db = _sum_to(dout, b.shape)
        dx = _matmul(dout, w.data.T)
        dw = _matmul(x.data.T, dout)
        return dx, dw, db


def linear(x, w, b) -> ITensor:
    return Linear()(x, w, b)


# ==========================================================================
# 神经网络常用算子
# ==========================================================================
