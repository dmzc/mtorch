r"""手写数字识别
用mlp网络解决手写数字识别，数据集为mnist
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from pathlib import Path

from mtorch.utils.data.dataloaders import DataLoader
from mtorch.utils.data.datasets import Mnist
from mtorch.utils.data.transforms import Compose, Flatten, Normalize, ToFloat32
from mtorch.optim import SGD
from mtorch.nn.modules import Sequential, Relu, Linear, CrossEntroyLoss
import time
import tracemalloc
from mtorch.utils import dumps
from mtorch import CACHE_DIR
from mtorch.utils.perf import CodeExecutionProfiler, MemoryUsageProfiler


# 超参数
batch_size = 100
hidden_size = 1000
max_epoch = 200
lr = 0.2


data_transform = Compose(Flatten(), Normalize(min=0, max=255), ToFloat32())

train_dataset = Mnist(
    dataset_type="train",
    data_transform=data_transform,
)
test_dataset = Mnist(
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

perf_infos = {}
tracemalloc.start()

o_count = 0
o_logs = {}
for idx in range(max_epoch):
    loop_start_mm, _ = tracemalloc.get_traced_memory()
    # loop_start_snapshot = tracemalloc.take_snapshot()
    loop_start_time = time.perf_counter_ns()
    count = 0
    o_log = {}
    o_logs[f"轮数{o_count}->{o_count+1}"] = o_log
    o_count = o_count + 1

    # with CodeExecutionProfiler(desc=f"{idx}轮训练耗时"):
    for item in train_loader:
        logs = []
        with (
            CodeExecutionProfiler(
                desc="数据加载耗时", logs=logs, start_time=loop_start_time
            ),
            MemoryUsageProfiler(
                desc="数据加载内存变化",
                logs=logs,
                start_current=loop_start_mm,
                # snap_before=loop_start_snapshot,
            ),
        ):
            pass
        data = item.data
        label = item.label

        with (
            CodeExecutionProfiler(desc="前向传播", logs=logs),
            MemoryUsageProfiler(desc="前向传播", logs=logs),
        ):
            y_pred = model.forward(data)
            loss = losser.forward(x=y_pred, t=label)
        # with (
        #     CodeExecutionProfiler(desc="反向传播", logs=logs),
        #     MemoryUsageProfiler(desc="反向传播", logs=logs),
        # ):
        model.clear_grads()
        loss.backward()
        with (
            CodeExecutionProfiler(desc="梯度更新", logs=logs),
            MemoryUsageProfiler(desc="梯度更新", logs=logs),
        ):
            optimizer.step()
        loop_start_time = time.perf_counter_ns()
        loop_start_mm, _ = tracemalloc.get_traced_memory()
        # loop_start_snapshot = tracemalloc.take_snapshot()
        o_log[f"{count}->{count+1}"] = logs
        count += 1

dumps(file=CACHE_DIR / f"stats/{time.asctime()}/perf_info", obj=o_logs)
