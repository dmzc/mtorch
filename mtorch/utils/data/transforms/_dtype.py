from mtorch._interfaces import ITransform
import numpy as np


class AsType(ITransform):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.astype(x, self.get_type())
        pass

    def get_type(self):
        raise NotImplementedError("Sub class must implemented get_type!")


# ========== 有符号整数 signed int ==========
class ToInt8(AsType):
    def get_type(self):
        return np.int8

    def __repr__(self):
        return "转int8(有符号8位整数)"


class ToInt16(AsType):
    def get_type(self):
        return np.int16

    def __repr__(self):
        return "转int16(有符号16位整数)"


class ToInt32(AsType):
    def get_type(self):
        return np.int32

    def __repr__(self):
        return "转int32(有符号32位整数)"


class ToInt64(AsType):
    def get_type(self):
        return np.int64

    def __repr__(self):
        return "转int64(有符号64位整数)"


# ========== 无符号整数 unsigned int ==========
class ToUInt8(AsType):
    def get_type(self):
        return np.uint8

    def __repr__(self):
        return "转uint8(无符号8位整数)"


class ToUInt16(AsType):
    def get_type(self):
        return np.uint16

    def __repr__(self):
        return "转uint16(无符号16位整数)"


class ToUInt32(AsType):
    def get_type(self):
        return np.uint32

    def __repr__(self):
        return "转uint32(无符号32位整数)"


class ToUInt64(AsType):
    def get_type(self):
        return np.uint64

    def __repr__(self):
        return "转uint64(无符号64位整数)"


# ========== 浮点数 float ==========
class ToFloat16(AsType):
    def get_type(self):
        return np.float16

    def __repr__(self):
        return "转float16(半精度浮点)"


class ToFloat32(AsType):
    def get_type(self):
        return np.float32

    def __repr__(self):
        return "转float32(单精度浮点)"


class ToFloat64(AsType):
    def get_type(self):
        return np.float64

    def __repr__(self):
        return "转float64(双精度浮点)"


# ========== 复数 complex ==========
class ToComplex64(AsType):
    def get_type(self):
        return np.complex64

    def __repr__(self):
        return "转complex64(32位复数)"


class ToComplex128(AsType):
    def get_type(self):
        return np.complex128

    def __repr__(self):
        return "转complex128(64位复数)"


# ========== 布尔 bool ==========
class ToBool(AsType):
    def get_type(self):
        return np.bool_

    def __repr__(self):
        return "转bool(布尔)"
