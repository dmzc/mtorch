from __future__ import annotations
from abc import abstractmethod
import math
import numpy as np
import weakref
from mtorch.config import ENABLE_BACKPROGATION
from mtorch.interfaces import ITensor, IOperator
from mtorch.tensor import Tensor


# ==========================================================================
# 算子基类
# ==========================================================================
class Operator(IOperator):

    def __init__(self):
        self.label = self.__class__.__name__
        self.inputs = None
        self.outputs = None
        self.generation = None

    def __call__(self, *xs: tuple[any]) -> list[ITensor] | ITensor:
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

    def forward(self, *xs: any) -> any:
        raise NotImplementedError

    def backward(self, dout: ITensor) -> any:
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
    底层基于 np.ndarray，前向计算遵循 NumPy 广播规则：
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

    def forward(self, x1: np.ndarray, x2: np.ndarray):
        self.__x1_shape = x1.shape
        self.__x2_shape = x2.shape

    def backward(self, dout: ITensor) -> list[ITensor]:
        dx1, dx2 = self.get_gradient(dout=dout)
        x1_shape = self.__x1_shape
        x2_shape = self.__x2_shape
        if x1_shape != x2_shape:
            return sum_to(dx1, x1_shape), sum_to(dx2, x2_shape)
        return dx1, dx2

    @abstractmethod
    def get_gradient(self, dout: ITensor) -> tuple[ITensor]: ...


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

    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        super().forward(x1, x2)
        return x1 + x2

    def get_gradient(self, dout):
        return dout, dout


def add(x1: any, x2: any) -> ITensor:
    """
    x1、x2 - IVariable、np.ndarray、数字 反正都会包装成IVariable
    """
    return Add()(x1, x2)


class Neg(Operator):
    """
    加法逆元
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: ITensor) -> ITensor:
        return -gy


def neg(x: any) -> ITensor:
    return Neg()(x)


class Mul(UrayOperator):
    """
    乘法
    """

    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        super().forward(x1=x1, x2=x2)
        return x1 * x2

    def get_gradient(self, dout):
        x0, x1 = self.inputs[0], self.inputs[1]
        return x1 * dout, x0 * dout


def mul(x1: any, x2: any):
    return Mul()(x1, x2)


class Sub(UrayOperator):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        super().forward(x1=x0, x2=x1)
        y = x0 - x1
        return y

    def get_gradient(self, dout: ITensor) -> list[ITensor]:
        return dout, -dout


def sub(x0: any, x1: any):

    # 减法、除法还是在拆解为加法、乘法逆元
    #
    # return x0-Neg(x1)
    return Sub()(x0, x1)


def rsub(x0: any, x1: any):
    return Sub()(x1, x0)


class Div(UrayOperator):
    """
    除法
    """

    def forward(self, x0: np.ndarray, x1: np.ndarray):
        super().forward(x1=x0, x2=x1)
        return x0 / x1

    def get_gradient(self, dout: ITensor):
        x0, x1 = self.inputs[0], self.inputs[1]
        gx0 = dout / x1
        gx1 = dout * (-x0 / x1**2)
        return gx0, gx1


def div(x0: any, x1: any):
    return Div()(x0, x1)


def rdiv(x0: any, x1: any):
    return Div()(x1, x0)


class Pow(Operator):
    """
    乘幂
    """

    def __init__(self, c: int):
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x**self.c
        return y

    def backward(self, gy: ITensor) -> ITensor:
        x = self.inputs[0]
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x: any, c: int):
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def backward(self, dout: ITensor) -> ITensor:
        return dout * cos(self.inputs[0])


def sin(x: any) -> ITensor:
    return Sin()(x)


def maclaurin_sin(x: ITensor, threshold=0.0001) -> ITensor:
    """
    麦克劳林展开求sin
    """
    y = 0
    for i in range(100000):
        const: int = 2 * i + 1
        c: float = (-1) ** i / math.factorial(const)
        t: ITensor = c * (x**const)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


class Cos(Operator):
    """
    余弦
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    def backward(self, dout: ITensor) -> ITensor:
        return dout * -sin(self.inputs[0])


def cos(x) -> ITensor:
    return Cos()(x)


class Tanh(Operator):
    """
    双曲正切
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        return y

    def backward(self, gy: ITensor) -> ITensor:
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x: any) -> ITensor:
    return Tanh()(x)


class Exp(Operator):
    """
    指数函数
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.exp(x)
        return y

    def backward(self, gy: ITensor) -> ITensor:
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Operator):
    """
    对数函数
    """

    def forward(self, x: np.ndarray):
        y = np.log(x)
        return y

    def backward(self, gy: ITensor) -> ITensor:
        x = self.inputs[0]
        gx = gy / x
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.__o_shape = x.shape
        return np.reshape(x, self.__n_shape)

    def backward(self, dout: ITensor) -> ITensor:
        return reshape(dout, self.__o_shape)


def reshape(x: np.ndarray | ITensor | list[int], shape: tuple[int]) -> ITensor:
    """
    list[int]代表原生多维数组
    """
    return Reshape(shape)(x)


class Transpose(Operator):
    """
    np.transpose:数据本身在内存中不会变,只是改变shape、stride。

    如果不传递参数，那么就是全部倒序下：

    x.shape # (2, 3, 4)

    x = x.transpose()

    x.shape # (4, 3, 2)
    """

    __n_axes: tuple[int]

    def __init__(self, n_axes=None):
        super().__init__()
        self.__n_axes = n_axes

    def forward(self, x: np.ndarray):
        return x.transpose(self.__n_axes)

    def backward(self, dout: ITensor) -> ITensor:
        n_axes = self.__n_axes
        if n_axes is None:
            return transpose(dout)
        axis_len = len(n_axes)
        # TODO:这里没搞懂，要连通np.transpose的算法一起搞清楚
        inv_axis = tuple(np.argsort([ax % axis_len for ax in n_axes]))
        return transpose(dout, inv_axis)


def transpose(x: np.ndarray | ITensor | list[int], axes=None):
    """
    list[int]代表原生多维数组
    """
    return Transpose(axes)(x)


# ==========================================================================
# 张量形状算子
# ==========================================================================

# ==========================================================================
# 常用张量算子
# ==========================================================================


class Sum(Operator):
    __keepdims: bool
    __axes: tuple[int] | int
    __from_shape: tuple[int]

    def __init__(self, keep_dims: bool = False, axes: int | tuple[int] | None = None):
        super().__init__()
        self.__keepdims = keep_dims
        if axes is not None:
            self.__axes = tuple(axes)
        else:
            self.__axes = None

    def forward(self, x: np.ndarray) -> int | float:
        self.__from_shape = x.shape
        return np.sum(x, axis=self.__axes, keepdims=self.__keepdims)

    def backward(self, dout: ITensor) -> ITensor:
        keepdims = self.__keepdims
        axes = self.__axes
        from_shape = self.__from_shape
        if keepdims:  # 正向传播时求和维度的没有被删除，直接reshape即可
            return broadcast_to(dout, from_shape)
        if axes is None:  # 所有轴求和，dout此时是一个标量，直接广播即可
            return broadcast_to(dout, from_shape)
        to_shape: list[int] = list(dout.shape)
        # 补全被删去的求和维度
        # 比如：原来（2，3，4，5，6)，按轴（1，3）求和得到（2，4，6）
        # 还原时需要先还原小索引，这样才不至于破坏后面的所有
        for axis in sorted(axes):
            to_shape.insert(axis, 1)
        return broadcast_to(reshape(dout, tuple(to_shape)), from_shape)


def sum(x: any, axes: tuple[int] | int = None, keepdims=False) -> ITensor:

    return Sum(axes=axes, keep_dims=keepdims)(x)


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

    def forward(self, x: np.ndarray) -> np.ndarray:
        from_shape = self.__from_shape = x.shape
        if from_shape == self.__to_shape:
            return x
        return np.broadcast_to(x, self.__to_shape)

    def backward(self, dout: ITensor) -> ITensor:
        return sum_to(dout, self.__from_shape)


def broadcast_to(x: np.ndarray, shape: tuple[int]) -> ITensor:
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

    def _raise_invalid_ndim_error(shape: tuple[int], to_shape: tuple[int]) -> None:
        raise ValueError(
            f"不符合SumTo的形状规则：\n原始形状：{shape}\n目标形状：{to_shape}"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:

        f_shape = self.__from_shape = x.shape
        t_shape = self.__to_shape
        if f_shape == t_shape:
            return x
        f_ndim = len(f_shape)
        t_ndim = len(t_shape)
        diff = f_ndim - t_ndim
        if diff < 0:
            self._raise_invalid_ndim_error()

        sum_axes: list[int] = list(range(diff))  # 那些轴要求和，前面多的维度肯定要求和

        for f_index in range(diff, f_ndim):
            t_dim = t_shape[f_index - diff]
            if f_shape[f_index] == t_dim:
                continue
            if t_dim == 1:
                sum_axes.append(f_index)
                continue
            # 目标维既不是1，也不相等，无法处理
            self._raise_invalid_ndim_error()
        result: np.ndarray = np.sum(x, axis=tuple(sum_axes), keepdims=True)
        if diff > 0:  # 前面多的维度被压缩了，要裁剪掉
            result = result.squeeze()
        return result

    def backward(self, dout: ITensor) -> ITensor:
        return broadcast_to(dout, self.__from_shape)


def sum_to(x: np.ndarray, shape: tuple[int]) -> ITensor:
    return SumTo(shape=shape)(x)


class Dot(Operator):

    def forward(self, x: np.ndarray, w: np.ndarray):
        return x.dot(w)

    def backward(self, dout: ITensor) -> ITensor:
        x, w = self.inputs
        return dot(dout, w.data.T), dot(x.data.T, dout)


def dot(x: np.ndarray, w: np.ndarray) -> ITensor:
    return Dot()(x, w)


# ==========================================================================
# 常用张量算子
# ==========================================================================


# ==========================================================================
# 损失算子
# ==========================================================================


class MeanSquareLoss(Operator):
    def forward(self, y_actual: np.ndarray, y_expect: np.ndarray) -> np.ndarray:
        diff: np.ndarray = y_actual - y_expect
        return np.sum((diff) ** 2) / diff.shape[0]

    def backward(self, dout: ITensor) -> ITensor:
        y_acutal, y_expect = self.inputs
        diff: ITensor = y_acutal - y_expect
        dy_actual = dout * 2 * diff / diff.shape[0]
        return dy_actual, -dy_actual


def mean_square_loss(y_actual, y_expect) -> ITensor:
    return MeanSquareLoss()(y_actual, y_expect)


# ==========================================================================
# 损失算子
# ==========================================================================


# ==========================================================================
# 激活算子
# ==========================================================================


class Sigmoid(Operator):
    def forward(self, x: np.ndarray) -> np.ndarray:
        # 传统形式：y=1/(1+np.exp(-x)),做变量代换，令z=x*0.5，两者是等价的。
        # tanh 在 numpy 底层做了稳定算法，不会出现指数直接爆炸计算，大幅度缓解极
        # 端输入下的溢出问题。
        return np.tanh(x * 0.5) * 0.5 + 0.5

    def backward(self, dout: ITensor) -> ITensor:
        x = self.outputs[0]()
        return dout * x * (1 - x)


def sigmoid(x) -> ITensor:
    return Sigmoid()(x)


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

    def forward(self, x: np.ndarray, w: np.ndarray, b: np.ndarray):
        out = x.dot(w)
        if b is not None:
            out = out + b
        return out

    def backward(self, dout) -> tuple[ITensor]:
        x, w, b = self.inputs
        db = None
        if b is not None:
            db = sum_to(dout, b.shape)
        dx = dot(dout, w.data.T)
        dw = dot(x.data.T, dout)
        return dx, dw, db


def linear(x, w, b) -> ITensor:
    return Linear()(x, w, b)


# ==========================================================================
# 神经网络常用算子
# ==========================================================================
