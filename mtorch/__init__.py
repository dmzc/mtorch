from mtorch._interfaces import ITensor, IDataset, IDataLoader, DatasetData, ITransform
from mtorch.core import Tensor, ENABLE_BACKPROGATION, pkg_enabled, operator

__version__ = "0.0.13"


def _setup():
    import mtorch.core.operator as F

    Tensor.__add__ = F.add
    Tensor.__radd__ = F.add
    Tensor.__mul__ = F.mul
    Tensor.__rmul__ = F.mul
    Tensor.__neg__ = F.neg
    Tensor.__sub__ = F.sub
    Tensor.__rsub__ = F.rsub
    Tensor.__truediv__ = F.div
    Tensor.__rtruediv__ = F.rdiv
    Tensor.__pow__ = F.pow
    Tensor.__matmul__ = F.matmul
    Tensor.__rmatmul__ = F.rmatmul
    Tensor.__imatmul__ = F.imatmul
    Tensor.__getitem__ = F.get_item


_setup()
__all__ = [
    "ITensor",
    "Tensor",
    "IDataset",
    "IDataLoader",
    "ITransform",
    "ENABLE_BACKPROGATION",
    "pkg_enabled",
    "operator",
    "DatasetData",
]
