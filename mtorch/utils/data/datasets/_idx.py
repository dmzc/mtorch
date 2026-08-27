from __future__ import annotations

import gzip
import struct
import sys
from io import BufferedReader
from pathlib import Path

import numpy as np

from mtorch import IDataset, CACHE_DIR
from typing import Any


class IDX:
    r"""
    IDX 文件二进制格式规范（LeCun MNIST 使用）
    全部32位整数均采用大端序(MSB优先)

    1. 魔数：4字节
        字节0、字节1：固定为 0x00
        字节2：数据类型码
            0x08 → uint8  无符号1字节整型
            0x09 → int8   有符号1字节整型
            0x0B → int16  有符号2字节整型
            0x0C → int32  有符号4字节整型
            0x0D → float32 单精度浮点数
            0x0E → float64 双精度浮点数
        字节3：维度数量 ndim，取值范围 1~255

    2. 维度大小列表：共 ndim 个 4字节大端无符号整数
        依次存储每一维的长度，例如图片集 [样本数, 高度, 宽度]

    3. 二进制数据区：
        行优先(C‑style)连续存储所有样本原始字节，无对齐填充、无分隔符。
    """

    _offset: int  # 元素起始偏移
    _image_size: int  # 一个图片占的字节数
    _image_count: int  # 图片个数
    _image_shape: tuple[int]  # 图片形状 ，例如训练集图片 [28,28]
    _type_code: int  # 元素数据类型：0x08=uint8，0x09=int8，0x0B=int16，0x0C=int32，0x0D=float32，0x0E=float64

    _fd: BufferedReader

    _initialized: bool

    _all_data: bool

    _the_all_data: np.ndarray

    def __init__(self, fd: BufferedReader, all_data: bool = False):
        self._fd = fd
        self._initialized = False
        self._offset = None
        self._type_code = None
        self._image_count = None
        self._image_shape = None
        self._image_size = None
        self._all_data = all_data
        self._the_all_data = None

    def read(self, index: int | list[int] | None = None) -> np.ndarray:
        r"""
        返回数据第一维始终为数量维度（即使只读一个），以灰度图片为例: (N, H, W)。

        Args:
            index: 要读取数据的索引，不传递就是读取所有数据。
        """

        self._init_header()

        indices: list[int] = None

        if index is not None:
            if isinstance(index, int):
                indices = [index]
            else:
                indices = index

        (dtype_size, dtype) = self._get_dtype(self._type_code)
        dtype_size = dtype_size * 8

        if self._all_data:
            if self._the_all_data is None:
                full_bytes = bytearray()
                self._fd.seek(self._offset)
                full_bytes = self._fd.read(self._image_count * self._image_size)
                np_arr = np.frombuffer(full_bytes, dtype=dtype)
                if sys.byteorder == "little" and dtype_size > 8:  # 是否小端序且单字节
                    np_arr = np_arr.byteswap()
                self._the_all_data = np_arr.reshape(
                    (self._image_count, *self._image_shape)
                )

            if indices is None:
                return self._the_all_data[:]
            else:
                return self._the_all_data[indices]

        count = self._image_count
        data_offset = self._offset

        image_size = self._image_size

        all_data = True  # 索引是否包含全部数据
        position_map: list[tuple[int, int]] = None
        ret_count = count  # 数据量
        if indices is not None:
            ret_count = len(indices)
            # 1. 数组越界、顺序检查
            last_idx = None
            idxs = set()
            for ordered_index, idx in enumerate(indices):
                if not isinstance(idx, int):
                    raise TypeError(
                        "Only 1‑D index arrays are supported; multi‑dimensional indices are not allowed."
                    )

                if idx < 0 or idx >= count:
                    raise ValueError(
                        f"Index {idx} out of range, total elements: {count}"
                    )

                idxs.add(idx)

                if position_map is not None:
                    position_map.append((ordered_index, idx))
                else:
                    if last_idx is None:
                        last_idx = idx
                    else:
                        o_last_idx = last_idx
                        last_idx = idx
                        if o_last_idx > last_idx:
                            position_map = []
                            for p_idx in range(ordered_index + 1):
                                position_map.append((p_idx, indices[p_idx]))

            all_data = len(idxs) == count
        else:
            all_data = True

        # 传入索引顺序(5, 2, 7)
        disk_indices = None  # 读磁盘顺序 (2, 5, 7)
        result_indices = None  # 磁盘顺序转返回结果顺序(1, 0, 2)
        if all_data:
            full_bytes = bytearray()
            self._fd.seek(data_offset)
            full_bytes = self._fd.read(self._image_count * image_size)
            np_arr = np.frombuffer(full_bytes, dtype=dtype)
        else:
            if position_map is not None:
                """
                如果索引不是全部数据且不是升序，则此变量记录磁盘读取顺序和返回顺序。
                例如：索引[5, 2, 7]，中间结果[(0, 5), (1, 2), (2, 7)]，此处按磁盘读取顺序排序[(1, 2), (0, 5), (2, 7)]
                """
                position_map.sort(key=lambda x: x[1])
                disk_indices = [dst_idx for _, dst_idx in position_map]
                result_indices = [src_idx for src_idx, in position_map]
            else:
                disk_indices = indices

            all_bytes = bytearray()
            # TODO: len(set(indices)) / count > 0.9  如果要读9成以上元素，不要一个个循环去读
            # TODO: 不要一个个去read，最好按磁盘页在内存中的缓存去读
            for idx in disk_indices:
                offset = data_offset + idx * image_size
                self._fd.seek(offset)
                all_bytes.extend(self._fd.read(image_size))
            np_arr = np.frombuffer(all_bytes, dtype=dtype)

        if sys.byteorder == "little" and dtype_size > 8:  # 是否小端序且单字节
            np_arr = np_arr.byteswap()

        np_arr = np_arr.reshape((ret_count, *self._image_shape))
        if result_indices is None:  # 读取全量数据或传入索引顺序本身是升序
            return np_arr
        else:
            out = np.empty_like(np_arr)
            out[result_indices] = np_arr
            return out

    @property
    def count(self) -> int:
        self._init_header()
        return self._image_count

    def _get_dtype(self, type_code: int) -> tuple[int, Any]:

        dtypes = {
            0x08: np.uint8,  # unsigned byte
            0x09: np.int8,  # signed byte
            0x0B: np.int16,  # short / int16
            0x0C: np.int32,  # int / int32
            0x0D: np.float32,  # float f32
            0x0E: np.float64,  # double f64
        }
        if type_code not in dtypes:
            raise ValueError(f"Unsupported IDX type_code={type_code:#04x}")
        dtype = dtypes[type_code]
        return (np.dtype(dtype).itemsize, dtype)

    def _init_header(self) -> None:
        """
        从已打开的二进制文件描述符解析IDX头部
        :param fd: rb模式打开的idx文件对象，调用后文件指针会被移动
        :return: IDXHeader实例
        :raises ValueError: 文件格式非法、截断、长度不匹配时抛出
        """
        if self._initialized:
            return
        self._initialized = True
        fd = self._fd

        fd.seek(0)
        # 读取最开头4字节魔数
        header4 = fd.read(4)
        if len(header4) != 4:
            raise ValueError("IDX文件头部不足4字节")

        # 按大端序解析4字节魔数字，>既是大端序，I转换为无符号32位整数
        magic: int = struct.unpack(">I", header4)[0]
        # 魔数高16位，标准IDX必须为0
        high16 = magic >> 16
        # 魔数第3字节：数据类型码
        self._type_code = type_code = (magic >> 8) & 0xFF
        # 魔数第4字节：维度个数
        ndim = magic & 0xFF

        if high16 != 0:
            raise ValueError(f"Invalid IDX magic number, high16={high16:#06x}")
        if not (1 <= ndim <= 255):
            raise ValueError(f"Invalid number of dimensions ndim={ndim}")

        (elem_bytes, _) = self._get_dtype(type_code=type_code)

        # 读取ndim个维度，每个维度占4字节大端int32
        dim_raw = fd.read(ndim * 4)
        if len(dim_raw) != ndim * 4:
            raise ValueError("IDX dimension header read incomplete, file truncated")

        # 解析得到各维度尺寸
        dims = list(struct.unpack(">" + "I" * ndim, dim_raw))
        self._image_count = dims[0]
        self._image_shape = tuple(dims[1 : len(dims)])
        # 头部总字节长度 = 魔数4字节 + ndim*4字节维度
        self._offset = data_offset = 4 + ndim * 4

        # 获取文件总字节数，用于完整性校验
        fd.seek(0, 2)
        file_total = fd.tell()
        fd.seek(0)

        _image_size = elem_bytes  # 一张图片占的字节数
        for dim in self._image_shape:
            _image_size *= dim

        self._image_size = _image_size

        # 数据区理论字节大小
        expected_data_bytes = self._image_size * self._image_count
        # 文件总大小必须等于头部+数据区，检测截断或者尾部多余垃圾字节
        if file_total != data_offset + expected_data_bytes:
            raise ValueError(
                f"IDX file length validation failed; "
                f"expected total bytes {data_offset + expected_data_bytes}, got {file_total}"
            )

    def close(self):
        if self._fd:
            self._fd.close()

    def __exit__(self):
        self.close()
        return False


class IDXDataset(IDataset):
    r"""
    Args:
        file: idx格式文件路径，支持gzip压缩格式与未压缩原始idx文件。
        work_dir: 工作目录；若输入为gzip压缩文件，会将文件解压至此目录；
                  为None时默认使用CACHE_DIR。

    Raises:
        ValueError: 文件不存在、文件格式不对、工作目录不可用。
    """

    _file: Path

    _work_dir: Path

    _idx: IDX

    _is_all_data: bool

    _all_data: np.ndarray

    def __init__(self, file: Path, work_dir: Path, all_data=False):
        super().__init__()
        if not file.exists() or not file.is_file():
            raise ValueError("File must be a idx format!")
        self._file = file
        self._work_dir = work_dir

        self._idx = None
        # TODO：BufferReader本来就不是为了用来做随机seek的，增量seek读文件，坑比较，先全量
        # TODO: 复现，连续读11张，然后在反过来读就不行了
        self._all_data = True

    def _load(self) -> None:
        if self._idx is not None:
            return
        fd: BufferedReader

        try:
            fd = open(self._file, "rb")
            header = fd.read(3)
            # 魔数 + deflate算法标记
            if header[0:2] == b"\x1f\x8b" and header[2] == 0x08:  # gzip文件
                t_fd = fd
                fd.seek(0)
                r_fd = open(self._decompress(fd), "rb")
                fd = r_fd
                t_fd.close()
        except Exception as e:
            if fd is not None:
                fd.close()
            raise e

        self._idx = IDX(fd=fd, all_data=self._all_data)

    def _decompress(self, fd: BufferedReader) -> Path:
        work_dir = self._work_dir
        if work_dir is None:
            work_dir = CACHE_DIR
        dst_file = work_dir / "~tmp"
        with (
            open(dst_file, "wb") as dst_fd,
            gzip.GzipFile(fileobj=fd, mode="rb") as src_fd,
        ):
            buf_size = 1024
            while chunk := src_fd.read(buf_size):
                dst_fd.write(chunk)
        return dst_file

    def __getitem__(self, indexes: int | list[int]) -> np.ndarray:
        self._load()
        return self._idx.read(indexes)

    def __len__(self) -> int:
        self._load()
        return self._idx.count

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        r"""
        此数据集会持有打开的文件句柄，存在文件资源泄漏风险。
        **推荐使用 `with` 语句管理实例生命周期**，离开 with 作用域时会自动调用 close() 释放句柄。
        如果不使用 with 上下文，则使用完毕后**必须手动调用 `.close()`** 释放文件句柄；
        仅依靠对象垃圾回收 __del__ 作为兜底，回收时机不可预测，不能作为正常释放手段。
        """
        if self._idx is not None:
            self._idx.close()
