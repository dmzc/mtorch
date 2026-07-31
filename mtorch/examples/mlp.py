import numpy as np
import matplotlib.pyplot as plt
from mtorch import Sequential, Linear, Sigmoid, SGD, MeanSquareLoss, ITensor

np.random.seed(0)

x = np.random.rand(100, 1)

y = np.sin(np.pi * 2 * x) + np.random.rand(100, 1) + 2


iters = 10000

model = Sequential(Linear(1, 10), Sigmoid(), Linear(10, 1))

optimzer = SGD(model, lr=0.2)
losser = MeanSquareLoss()


for index in range(iters):
    y_pred = model.forward(x)
    loss = losser.forward(y_actual=y_pred, y_expect=y)
    model.clear_grads()
    loss.backward()
    optimzer.step()
    print(f"第{index}轮损失{loss.data}")


plt.scatter(x, y, s=10)
plt.xlabel("x")
plt.ylabel("y")
t = np.arange(0, 1, 0.001)[:, np.newaxis]
y_pred: ITensor = model.forward(t)
plt.plot(t, y_pred.data, color="r")
plt.show()
