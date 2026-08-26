from mtorch._interfaces import ITransform
import numpy as np


class Normalize(ITransform):
    r"""Min‑Max 0‑1归一化（Normalization）

    Args:
        min:最小值
        max:最大值
    不传递则从传递的数据中推测最大/最小值（此时必须是全量数据）。

    公式：x' = (x - x_min) / (x_max - x_min)
    将数据线性缩放到 [0, 1]。
    二维数组约定：每行是样本，每列是特征，按列计算min/max。

    适用场景：
        1. 距离类算法：KNN、K‑Means，消除量纲对距离计算的影响；
        2. 神经网络输入：Sigmoid激活网络、图像像素(0‑255→0‑1)；
        3. 梯度下降求解的线性模型，加速收敛；
        4. 多指标打分、多曲线可视化对比，需要固定输出区间。

    不适用场景：
        1. 存在显著离群点：min/max被异常值拉扯，数据被挤压；优先改用Z‑score；
        2. PCA主成分分析：Min‑Max会扭曲方差，推荐Z‑score标准化；
        3. 新样本容易超出训练集min/max范围，变换结果会溢出[0,1]；
        4. 树模型(决策树、随机森林、XGBoost等)：树分裂不受缩放影响，无需归一化。

    注意：
        1. 机器学习任务：fit统计min/max仅使用训练集，测试集仅做transform；
        2. 增加除零保护：特征全部取值相同时直接输出0，避免除以0报错。
    """

    _min: int | float
    _max: int | float

    def __init__(self, min: int | float = None, max: int | float = None):
        self._min = min
        self._max = max

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        x_min = self._min
        x_max = self._max
        if x_min is None:
            x_min = np.min(arr, axis=0)
        if x_max is None:
            x_max = np.max(arr, axis=0)
        delta = x_max - x_min

        # 保护：特征全部相同的时候，避免除0，输出全0
        # where: delta为0时取0，否则执行缩放公式
        scaled = np.where(delta > 1e-12, (arr - x_min) / delta, 0.0)
        return scaled

    def __repr__(self):
        return "0-1归一化"


class Standardize(ITransform):
    """
    z-score标准化
    """

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

    def __repr__(self):
        return "z-score标准化"
