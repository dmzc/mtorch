from __future__ import annotations
from pathlib import Path
from mtorch.core._env import pkg_enabled
import numpy as np
from typing import Iterable, BinaryIO, Any


def _dump_item(obj: Any, orjson_enabled: bool = True) -> bytes:
    def _json_default_handler(obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    if orjson_enabled:
        import orjson

        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)
    else:
        import json

        return json.dumps(
            obj,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default_handler,
        ).encode("utf-8")


def dumps_stream(
    file: str | Path = None,
    iteratable_obj: Iterable = None,
    single: bool = False,
    compress: bool = False,
) -> Path:
    r"""
    与 ``dumps`` 功能语义一致，采用流式写入实现，避免完整对象驻留内存，降低内存峰值。

    ``path`` 为文件路径主体，**请勿携带文件后缀**，函数会自动追加对应扩展名；文件名部分禁止包含 "."。

    > 注意：``iteratable_obj`` 为迭代器，迭代产出元素支持 numpy 数组、list、tuple、int、float、str、bool，
    以及具备 ``__dict__`` 的自定义类实例。迭代每次产出的批量条目数量必须保持固定，该批量大小需要通过 ``per_size`` 指定。

    Args:
        path: 文件路径主体，不带后缀，扩展名由函数自动追加；文件名部分不能含 "."。
        iteratable_obj: 可迭代对象，迭代产出支持 numpy 数组、list、tuple、int、float、str、bool，
            以及具备 ``__dict__`` 的自定义类实例；每次迭代返回的批量条目数必须固定。
        per_size: iteratable_obj 每次迭代返回的数据条目数量，不指定时默认按单条（1）处理。
        compress: 是否开启压缩输出；优先 zstandard，库不可用时自动降级为 gzip。
     Returns:
        Path: 写入完成后的文件路径
     Raises:
        OSError: 文件目录创建失败、文件写入失败、磁盘空间不足、权限不足等 IO 相关错误。
        TypeError: 对象包含无法序列化的类型，不支持的自定义对象。
        ModuleNotFoundError: 开启压缩时，zstandard / gzip 依赖均不可用。
        ValueError: ``path`` 文件名部分包含 "."，违反命名约束。
    """
    file: Path = Path(file)
    parent_foleder = file.parent
    filename = file.name

    # 3. 确保目录存在
    if not parent_foleder.exists():
        parent_foleder.mkdir(parents=True, exist_ok=True)

    tmp_file = parent_foleder / "~tmp"

    ORJSON_ENABLED = pkg_enabled("orjson")

    def _do_write(io: BinaryIO):
        io.write(b"[")
        is_first = True
        is_single = single == 1
        for items in iteratable_obj:
            if not is_first:
                io.write(b",")
            is_first = False
            if is_single:

                io.write(_dump_item(items, orjson_enabled=ORJSON_ENABLED))
            else:
                is_item_first = True
                for item in items:
                    if not is_item_first:
                        io.write(b",")
                    is_item_first = False
                    io.write(_dump_item(item, orjson_enabled=ORJSON_ENABLED))
        io.write(b"]")

    try:

        with open(tmp_file, "wb") as t_f:
            if not compress:
                file = parent_foleder / f"{filename}.json"
                _do_write(t_f)
            elif pkg_enabled("zstandard"):

                import zstandard

                file = parent_foleder / f"{filename}.zst"
                with zstandard.ZstdCompressor().stream_writer(
                    t_f, closefd=False
                ) as z_f:
                    _do_write(z_f)

            else:
                import gzip

                file = parent_foleder / f"{filename}.gzip"
                with gzip.GzipFile(fileobj=t_f, mode="wb") as g_f:
                    _do_write(g_f)
            t_f.flush()
            # 底层import，但是没有提示，嵌套层级太多，IDE的bug
            import os

            os.fsync(t_f.fileno())
        import os

        os.replace(tmp_file, file)
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise
    return file


def dumps(
    file: str | Path,
    obj: list | tuple | int | float | str | bool | np.ndarray,
    compress=False,
) -> Path:
    r"""
    将内存对象 ``obj`` 序列化为 JSON 文件。

    当 ``compress=True`` 开启压缩输出：优先使用 zstandard；zstandard 不可用时自动回退至 gzip。

    ``path`` 为文件路径主体，**不要携带文件后缀**，函数会自动追加对应扩展名；文件名部分禁止包含 "."。

    > 注意：该接口对完整驻留内存的对象执行一次性序列化与压缩，**不支持流式序列化、流式压缩**。
    仅适用于中小规模对象；处理大对象请使用 ``dumps_stream`` 流式接口，避免内存峰值导致 OOM。

    本接口未做数值精度适配，**不适合存储权重等对数值精度敏感的数据**，主要面向语料类业务数据。
    语料场景下浮点数大多属于观测、统计、特征类数值，本身自带业务噪声；float32 的 7 位有效数字通常可以覆盖绝大多数业务的噪声水平。

    Args:
        path: 文件路径主体，不带后缀，扩展名由函数自动追加。
        obj: 待序列化内存对象，支持 numpy 数组、list、tuple、int、float、str、bool，以及具备 ``__dict__`` 的自定义类实例。
        compress: 是否开启压缩输出；优先 zstandard，依赖缺失时自动降级为 gzip。

    Returns:
        Path: 写入完成后的文件完整路径。

    Raises:
        OSError: 文件目录创建失败、文件写入失败、磁盘空间不足、权限不足等 IO 相关错误。
        TypeError: 对象包含无法序列化的类型，不支持的自定义对象。
        ModuleNotFoundError: 开启压缩时，zstandard / gzip 依赖均不可用。
        ValueError: ``path`` 文件名部分包含 "."，违反命名约束。
    """

    file: Path = Path(file)
    parent_foleder = file.parent
    filename = file.name

    # 1. 序列化
    json_bytes = _dump_item(obj=obj, orjson_enabled=pkg_enabled("orjson"))

    # 2. 压缩文件
    if not compress:  # .json
        file = parent_foleder / f"{filename}.json"
    else:
        if pkg_enabled("zstandard"):  # .zst
            import zstandard

            file = parent_foleder / f"{filename}.zst"

            json_bytes = zstandard.compress(json_bytes)
        else:  # .gzip
            import gzip

            file = parent_foleder / f"{filename}.gzip"
            json_bytes = gzip.compress(json_bytes)

    # 3. 确保目录存在
    if not parent_foleder.exists():
        parent_foleder.mkdir(parents=True, exist_ok=True)

    # 4. 写入，写入是原子的，但是有任何因素导致失败，都不会写入，也不会破坏原文件。
    try:
        import os

        tmp_file = file.with_suffix(".tmp")

        with open(tmp_file, "wb") as f:
            f.write(json_bytes)
            f.flush()
            os.fsync(f.fileno())

        """ os.replace
        封装抹平平台差异：
        POSIX：直接调用系统 rename()，覆盖已有普通文件，原子。
        Windows：内部调用 MoveFileEx 带 MOVEFILE_REPLACE_EXISTING，实现覆盖。
        统一行为：只要是普通文件，目标存在就覆盖；原子操作；要求 src、dst 必须在同一个文件系统（不能跨磁盘 / 跨挂载点）。
        如果 dst 是目录，会抛异常，不会覆盖目录
        """

        os.replace(tmp_file, file)
    except Exception as e:
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)
        raise e
    return file


def loads(file: str | Path) -> Any:
    r"""
    TODO：支持流式读大文件内容
    """
    file = Path(file)
    if not file.exists() and not file.is_file():
        raise ValueError(f"File not found:{str(file)}")
    file = Path(file)
    with open(file, "rb") as f:
        head = f.read(32)  # 只读取开头32字节，足够识别魔数
        f.seek(0)

        def assert_json_bytes(json_bytes: bytes) -> bytes:
            if not json_bytes.lstrip().startswith(
                (b"{", b"[")
            ):  ## 裸JSON，第一个有效字节是 { 或者 [
                raise ValueError(f"File is not a JSON:{str(file)}")
            return json_bytes

        json_bytes: bytes = None

        if head.startswith(b"\x28\xb5\x2f\xfd"):
            if not pkg_enabled("zstandard"):
                raise ValueError("Zstabdard Required!")
            import zstandard

            dctx = zstandard.ZstdDecompressor()
            try:
                data = f.read()
                json_bytes = dctx.decompress(data)
            except zstandard.ZstdError as e:
                # could not determine content size in frame header → 流式帧，走stream_reader
                if "could not determine content size in frame header" in str(e):
                    f.seek(0)
                    with dctx.stream_reader(f) as reader:
                        json_bytes = reader.read()
                else:
                    raise e

        elif head.startswith(b"\x1f\x8b"):
            import gzip

            json_bytes = gzip.decompress(f.read())
        else:
            json_bytes = f.read()

        if not json_bytes.lstrip().startswith(
            (b"{", b"[")
        ):  ## 裸JSON，第一个有效字节是 { 或者 [
            raise ValueError(f"File is not a JSON:{str(file)}")

        if pkg_enabled("orjson"):
            import orjson

            return orjson.loads(json_bytes)
        else:
            import json

            return json.loads(json_bytes.decode("utf-8"))
