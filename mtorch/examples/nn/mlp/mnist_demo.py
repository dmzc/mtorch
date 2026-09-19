r"""手写数字识别
用mlp网络解决手写数字识别，数据集为mnist
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from pathlib import Path
import time
from mtorch.utils.data.dataloaders import DataLoader
from mtorch.utils.data.datasets import Mnist
from mtorch.utils.data.transforms import Compose, Flatten, Normalize, ToFloat32
from mtorch.optim import SGD
from mtorch.nn.modules import Sequential, Relu, Linear, CrossEntroyLoss
from mtorch.utils.persist import PersistService
from mtorch import CACHE_DIR
from mtorch.utils.perf import CodeExecutionProfiler, MemoryUsageProfiler, ProfilerService


# 超参数
batch_size = 100
hidden_size = 1000
max_epoch = 2
lr = 0.2


data_transform = Compose(Flatten(), Normalize(min=0, max=255), ToFloat32())

train_dataset = Mnist(
    dataset_type="train",
    # data_transform=data_transform,
)
test_dataset = Mnist(
    dataset_type="test",
    # data_transform=data_transform,
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

o_count = 0
o_logs = {}
for idx in range(max_epoch):
    loop_start_py_mm = ProfilerService.get_task_py_memory()
    loop_start_all_mm = ProfilerService.get_task_all_memory()
    # loop_start_snapshot = tracemalloc.take_snapshot()
    loop_start_time = ProfilerService.get_current_time()
    count = 0
    o_log = {}
    o_logs[f"轮数{o_count}->{o_count+1}"] = o_log
    o_count = o_count + 1

    c_profiler = CodeExecutionProfiler(scene="")
    m_profiler = MemoryUsageProfiler(scene="")
    with CodeExecutionProfiler(scene=f"{idx}轮训练耗时"):
        for item in train_loader:
            logs = []
            c_profiler.set_consumer(logs)
            m_profiler.set_consumer(consumer=logs)
            with (
                c_profiler.set_start_time(loop_start_time).set_scene("数据加载"),
                m_profiler.set_start_am(loop_start_all_mm)
                .set_start_pm(loop_start_py_mm)
                .set_scene("数据加载"),
            ):
                pass
            data = item.data
            label = item.label
            c_profiler.set_start_time(None)
            m_profiler.set_start_am(None)
            m_profiler.set_start_pm(None)
            with (
                c_profiler.set_scene("前向传播"),
                m_profiler.set_scene(scene="前向传播"),
            ):
                y_pred = model.forward(data)
                loss = losser.forward(x=y_pred, t=label)
            with (
                c_profiler.set_scene("反向传播"),
                m_profiler.set_scene(scene="反向传播"),
            ):
                model.clear_grads()
                loss.backward()
            with (
                c_profiler.set_scene("梯度更新"),
                m_profiler.set_scene(scene="梯度更新"),
            ):
                optimizer.step()
            loop_start_time = ProfilerService.get_current_time()
            loop_start_py_mm = ProfilerService.get_task_py_memory()
            loop_start_all_mm = ProfilerService.get_task_all_memory()
            # loop_start_snapshot = tracemalloc.take_snapshot()
            o_log[f"{count}->{count+1}"] = logs
            count += 1

asctime = time.asctime()
asctime = asctime.replace(" ", "_").replace(":", "-")
PersistService.save_json(file=CACHE_DIR / f"stats/{asctime}/perf_info", obj=o_logs)
