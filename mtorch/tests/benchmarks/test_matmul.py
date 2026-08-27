import mtorch.core.operator as F
from mtorch import ITensor

import numpy as np

from mtorch.utils.perf import CodeExecutionProfiler


w = np.random.rand(784, 100)
b = np.random.rand(100)
x = np.random.rand(100, 784)
# profiler之前，预热，把数据拉入cache
_ = w.sum()
_ = b.sum()
_ = x.sum()
# 再跑 matmul backward

with CodeExecutionProfiler("矩阵算子前向"):
    out: ITensor = F.matmul(x, w) + b
    # x.dot(w)+b
# with CodeExecutionProfiler("矩阵算子反向"):
out.backward()
