import numpy as np
import matplotlib.pyplot as plt
from mtorch import Sequential, Linear, Sigmoid, SGD, MeanSquareLoss
from mtorch import UnivariateFunctionDataset, DataLoader


def test_mlp():

    # 超参数
    lr = 0.2
    epoch = 100
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
    first_loss: float = None
    last_loss: float = None
    for index in range(epoch):

        loss = None
        for x, y in dataloader:
            y_pred = model.forward(x)
            loss = losser.forward(y_actual=y_pred, y_expect=y)
            model.clear_grads()
            loss.backward()
            optimzer.step()

        if index == 0:
            first_loss = loss.data.tolist()
        if index == epoch - 1:
            last_loss = loss.data.tolist()
    assert last_loss < first_loss, "训练了100轮，损失函数正常下降了"
