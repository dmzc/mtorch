from mtorch._interfaces import (
    ITrainer,
    IModule,
    IDataLoader,
    IOptimizer,
    IMetric,
    DatasetData,
    ITensor,
)
from mtorch.nn.modules import CrossEntroyLoss, MeanSquareLoss
from typing import Literal
from pathlib import Path
import time


class Trainer(ITrainer):

    _model: IModule  # 模型

    _losser: IModule

    _train_dataloader: IDataLoader  # 训练数据加载器

    _optimizer: IOptimizer  # 梯度更新器

    _word_dir: Path  # 工作目录，存放工作日志，最优点权重

    _valid_dataloader: IDataLoader  # 验证数据加载器

    _valid_gap: int  # 每隔多少轮进行一次验证

    _mertrics: list[IMetric]  # 评估指标

    _max_epoch: int  # 最大轮

    # TODO:验证指标

    # TODO:性能指标

    def __init__(
        self,
        model: IModule,
        train_loader: IDataLoader,
        optimizer: IOptimizer,
        word_dir: Path,
        loss: Literal["mse", "crossEntroy"],
        max_epoch: int = 10,
        metrics: list[IMetric] = None,
        valid_loader: IDataLoader = None,
    ):
        self._model = model
        self._train_dataloader = train_loader
        self._optimizer = optimizer
        self._word_dir = (
            word_dir
            / f"train/{str(time.asctime()).replace(' ', '_').replace(':', '-')}"
        )
        self._valid_dataloader = valid_loader
        self._max_epoch = max_epoch
        self._mertrics = metrics
        if loss == "crossEntroy":
            self._losser = CrossEntroyLoss(axis=1)
        elif loss == "mse":
            self._losser = MeanSquareLoss()

    def train(self):
        model: IModule = self._model
        losser = self._losser
        optimizer = self._optimizer
        for idx in self._max_epoch:
            for item in self._train_dataloader:
                item: DatasetData = item
                data = item.data
                label = item.label
                label_pred = model.forward(data)
                loss: ITensor = losser.forward(label_pred, label)
                model.clear_grads()
                loss.backward()
                optimizer.step()

    def save(self, epoch: int, logs: any):
        content = {
            "max_epoch": self._max_epoch,
            "optimizer": str(self._optimizer),
            "current_epoch": epoch,
            "logs": logs,  # 训练日志，比如：每轮训练耗费时间、损失值、准确率
            "model": str(self._model),
            "loss": str(self._losser),
            "params": [],  # TODO:模型权重
        }
        pass

    def load(self, json):
        pass
