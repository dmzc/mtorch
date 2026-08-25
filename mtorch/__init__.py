from mtorch._interfaces import DatasetData, IDataLoader, IDataset, ITensor, ITransform
from mtorch.core import ENABLE_BACKPROGATION, ROOT_DIR, Tensor, operator, pkg_enabled

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
    "ENABLE_BACKPROGATION",
    "ROOT_DIR",
    "DatasetData",
    "IDataLoader",
    "IDataset",
    "ITensor",
    "ITransform",
    "Tensor",
    "operator",
    "pkg_enabled",
]
