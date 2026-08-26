import numpy as np
from mtorch.utils.data.transforms import (
    ToInt8,
    ToInt16,
    ToInt32,
    ToInt64,
    ToUInt8,
    ToUInt16,
    ToUInt32,
    ToUInt64,
    ToFloat16,
    ToFloat32,
    ToFloat64,
    ToComplex64,
    ToComplex128,
    ToBool,
)


def test_toInt8():
    t = ToInt8()
    arr = np.array([-128, 0, 127], dtype=np.int64)
    out = t(arr)
    assert out.dtype == np.int8
    np.testing.assert_array_equal(out, arr)


def test_toInt16():
    t = ToInt16()
    arr = np.array([-32768, 0, 32767], dtype=np.int64)
    out = t(arr)
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, arr)


def test_toInt32():
    t = ToInt32()
    arr = np.array([-2147483648, 0, 2147483647], dtype=np.int64)
    out = t(arr)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, arr)


def test_toInt64():
    t = ToInt64()
    arr = np.array([-9223372036854775808, 0, 9223372036854775807], dtype=np.int64)
    out = t(arr)
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, arr)


def test_toUInt8():
    t = ToUInt8()
    arr = np.array([0, 127, 255], dtype=np.int64)
    out = t(arr)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, arr)


def test_toUInt16():
    t = ToUInt16()
    arr = np.array([0, 32768, 65535], dtype=np.int64)
    out = t(arr)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, arr)


def test_toUInt32():
    t = ToUInt32()
    arr = np.array([0, 2147483648, 4294967295], dtype=np.int64)
    out = t(arr)
    assert out.dtype == np.uint32
    np.testing.assert_array_equal(out, arr)


def test_toUInt64():
    t = ToUInt64()
    arr = np.array([0, 9223372036854775808, 18446744073709551615], dtype=np.uint64)
    out = t(arr)
    assert out.dtype == np.uint64
    np.testing.assert_array_equal(out, arr)


def test_toFloat16():
    t = ToFloat16()
    arr = np.array([-1.5, 0.0, 2.5], dtype=np.float64)
    out = t(arr)
    assert out.dtype == np.float16
    np.testing.assert_allclose(out, arr)


def test_toFloat32():
    t = ToFloat32()
    arr = np.array([-1.5, 0.0, 2.5], dtype=np.float64)
    out = t(arr)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, arr)


def test_toFloat64():
    t = ToFloat64()
    arr = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    out = t(arr)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, arr)


def test_toComplex64():
    t = ToComplex64()
    arr = np.array([1 + 2j, 0 - 1j], dtype=np.complex128)
    out = t(arr)
    assert out.dtype == np.complex64
    np.testing.assert_allclose(out, arr)


def test_toComplex128():
    t = ToComplex128()
    arr = np.array([1 + 2j, 0 - 1j], dtype=np.complex64)
    out = t(arr)
    assert out.dtype == np.complex128
    np.testing.assert_allclose(out, arr)


def test_toBool():
    t = ToBool()
    arr = np.array([0, 1, -1, 100], dtype=np.int32)
    expect = np.array([False, True, True, True])
    out = t(arr)
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, expect)


def test_repr_all_dtype_transforms():
    """验证 __repr__ 不会抛异常，方便调试打印"""
    cases = [
        ToInt8(),
        ToInt16(),
        ToInt32(),
        ToInt64(),
        ToUInt8(),
        ToUInt16(),
        ToUInt32(),
        ToUInt64(),
        ToFloat16(),
        ToFloat32(),
        ToFloat64(),
        ToComplex64(),
        ToComplex128(),
        ToBool(),
    ]
    for inst in cases:
        s = repr(inst)
        assert isinstance(s, str)
        assert len(s) > 0
