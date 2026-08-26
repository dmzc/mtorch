r"""手写数字识别
用mlp网络解决手写数字识别，数据集为mnist
"""

from pathlib import Path

from mtorch.utils.data.dataloaders import DataLoader
from mtorch.utils.data.datasets import Mnist
from mtorch.utils.data.transforms import Compose, Flatten, Normalize, ToFloat32
from mtorch.optim import SGD
from mtorch.nn.modules import Sequential, Relu, Linear, CrossEntroyLoss

# 超参数
batch_size = 100
hidden_size = 1000
max_epoch = 20
lr = 0.2


data_transform = Compose(Flatten(), Normalize(min=0, max=255), ToFloat32())

train_dataset = Mnist(
    root_dir=Path(__file__).parent / "tmp",
    dataset_type="train",
    data_transform=data_transform,
)
test_dataset = Mnist(
    root_dir=Path(__file__).parent / "tmp",
    dataset_type="test",
    data_transform=data_transform,
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

model = Sequential(
    Linear(784, hidden_size=hidden_size),
    Relu(),
    Linear(input_size=hidden_size, hidden_size=10),
)
optimizer = SGD(params_obj=model, lr=lr)


losser = CrossEntroyLoss(axis=1)
flag = False
for index in range(max_epoch):

    import time
    import tracemalloc

    tracemalloc.start()
    # 迭代之前记录基准
    current_prev, peak_prev = tracemalloc.get_traced_memory()
    snapshot1 = tracemalloc.take_snapshot()
    if flag:
        break
    t1 = time.perf_counter()
    for item in train_loader:
        t2 = time.perf_counter()
        data = item.data
        label = item.label

        current_now, peak_now = tracemalloc.get_traced_memory()
        snapshot2 = tracemalloc.take_snapshot()

        delta = current_now - current_prev
        print(f"Python堆新增 {delta / 1024:.2f} KB  peak {peak_now /1024:.2f} KB")

        # 打印分配最多的5行
        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        for stat in top_stats[:5]:
            print(stat)

        y_pred = model.forward(data)
        t3 = time.perf_counter()
        loss = losser.forward(x=y_pred, t=label)
        t4 = time.perf_counter()
        model.clear_grads()
        loss.backward()
        t5 = time.perf_counter()
        optimizer.step()
        t6 = time.perf_counter()
        flag = True
        break
        # print(
        #     f"load:{(t2-t1)*1000:6.2f} ms | "
        #     f"forward:{(t3-t2)*1000:6.2f} ms | "
        #     f"loss:{(t4-t3)*1000:6.2f} ms | "
        #     f"backward:{(t5-t4)*1000:6.2f} ms | "
        #     f"step:{(t6-t5)*1000:6.2f} ms | "
        #     f"all:{(t6-t1)*1000:6.2f} ms"
        # )
    # print(f"第{index}轮损失{loss.data}")
