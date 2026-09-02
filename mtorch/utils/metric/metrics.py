from mtorch._interfaces import IMetric, DataArray
import numpy as np


class AccuracyMetric(IMetric):

    _total: int
    _correct: int

    def __init__(self):
        self._total = 0
        self._correct = 0

    def update(self, pred: DataArray, target: DataArray) -> IMetric:

        pred_cls = np.argmax(pred, axis=1)
        batch_correct = np.sum(pred_cls == target)
        batch_size = target.shape[0]

        self._correct += int(batch_correct)
        self._total += int(batch_size)
        return self

    def compute(self) -> float:
        return self._correct / self._total

    def reset(self) -> IMetric:
        self._total = self._correct = 0
        return self
