import numpy as np
import matplotlib.pyplot as plt

from mtorch import ITensor
from mtorch.optim import SGD
from mtorch.nn.modules import Sequential, Linear, Sigmoid, MeanSquareLoss
from mtorch.utils.data.datasets import UnivariateFunctionDataset
from mtorch.utils.data.dataloaders import DataLoader

r"""
mlp网络解决线性回归
"""

# 超参数
lr = 0.2
epoch = 10000
batch_size: int = None

model = Sequential(
    Linear(1, 10),
    Sigmoid(),
    Linear(10, 1),
)
optimzer = SGD(model, lr=lr)
losser = MeanSquareLoss()

dataloader = DataLoader(
    dataset=UnivariateFunctionDataset(
        func=lambda x: np.sin(np.pi * 2 * x) + 2, data_size=100
    ),
    batch_size=batch_size,
)
for index in range(epoch):
    for x, y in dataloader:
        y_pred = model.forward(x)
        loss = losser.forward(y_actual=y_pred, y_expect=y)
        model.clear_grads()
        loss.backward()
        optimzer.step()
        print(f"第{index}轮损失{loss.data}")

# 训练数据：展示为散点
figure, axes = plt.subplots()
axes.scatter(x, y, s=10)
axes.set_xlabel("x")
axes.set_ylabel("y")

# 预测值：展示为函数图像，函数图像为线条，所以x值连续密集，不能随机
t = np.arange(0, 1, 0.001)[:, np.newaxis]
y_pred: ITensor = model.forward(t)
axes.plot(t, y_pred.data, color="r")
plt.show()
