from __future__ import annotations
import numpy as np
from mtorch import Tensor, ITensor
import orjson
from pathlib import Path
import zstandard


class Object:
    _data: np.ndarray

    def __init__(self, data: np.ndarray):
        self._data = data

    def __getitem__(self, slices: tuple[int | any]):
        print(f"{slices}")
        return self._data[slices]

    def __setitem__(self, key, value):
        self._data[key] = value
        print(key)
        print(value)


obj = Object(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]))
# obj[0]  # __get_item__(obj, 0, None)
# obj[0, 1]  # __getitem__(obj, (0,1), None)
# obj[:, 1]  # __getitem__(obj, (slice(None, None, None), 1))
# obj[1:3, 1]  # __getitem__(obj, (slice(1, 3, None), 1))
# obj[1:3, 0:1]  # __getitem__(obj, (slice(1, 3, None), slice(0, 1, None)))
# obj[:2:, 1]  # __getitem__(obj,(slice(None, 2, None), 1))
# obj[1:, 1]  # (slice(1, None, None), 1)
# obj[:2, 1]  # (slice(None, 2, None), 1)
# obj[:2:4]  # slice(None, 2, 4)
# obj[1:4:1]  # slice(1, 4, 1)
# obj[0:2] = [[34, 45, 56], [78, 89, 999]]
obj._data = obj[[0, 2]]  # [0,2]
print(obj._data)
# fmt:off
tensor1 = Tensor([
    [1, 2, 3], 
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

tensor2=tensor1[2,1]
# print(tensor2)

def save_json(path:str|Path,obj,indent:bool=False,compress=False):
    path:Path= Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"wb") as f:
        opt=orjson.OPT_SERIALIZE_NUMPY
        opt|=orjson.OPT_SERIALIZE_DATACLASS
        if indent:
            opt|=orjson.OPT_INDENT_2
        json_bytes=orjson.dumps(obj,option=opt)
        if compress:
            cctx = zstandard.ZstdCompressor()
            compressed = cctx.compress(json_bytes)
            f.write(compressed)
        else:
            f.write(json_bytes)
def load_json(path:str|Path,decompress=False):
    path=Path(path)
    with open(path,"rb") as f:
        if decompress:
            dctx=zstandard.ZstdDecompressor()
            json_bytes=dctx.decompress(f.read())
            return orjson.loads(json_bytes)
        else:
            return orjson.loads(f.read())

# np1=np.random.rand(100,100)
# np1=np.array(np1)

# path=Path(__file__).parent/"data/orjson.json"
# save_json(path=path,obj=np1)
# data=load_json(path=path)

# compress_path=path=Path(__file__).parent/"data/orjson.zst"

# save_json(path=compress_path,obj=np1,compress=True)
# data=load_json(path=compress_path,decompress=True)
# print(data)

# from importlib.util import find_spec
# from importlib.metadata import version
# result=find_spec("orjson")
# print(result.name)
# print(version("orjson"))
# path=Path("c:/tes/ted.d/ted.DS")
# print("filename",path.name)
# with open(Path(__file__).parent/"data/test.json","wb") as f:

#     # 这里b代表是直接写字节，不是普通字符串
#     f.write(b"[")
#     f.write(orjson.dumps([12,23,34,45]))
#     f.write(b",")
#     f.write(orjson.dumps([12,34,54]))
#     f.write(b"]")
# with open(Path(__file__).parent/"data/test.json","wb") as f:
#     import json

#     obj=[[12,34],[12,34]]
#     f.write(json.dumps(obj).encode("utf-8"))
