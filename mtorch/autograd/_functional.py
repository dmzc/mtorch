from mtorch._interfaces import ITensor, IOperator, DataArray
from typing import Any, Optional
import numpy as np


def backward(tensor: ITensor) -> None:
    r"""
    反向传播入口函数，执行计算图的反向自动微分。
    从输出tensor开始，拓扑遍历计算图中的算子(creator)，累积各输入的梯度。

    1. 若tensor无creator(叶子节点、无计算历史)，直接返回，无需反向传播。
    2. 初始化tensor的输出梯度grad，作为反向传播的上游梯度信号。
    3. 收集所有待反向的算子，做拓扑排序，避免重复处理同一个算子(seen_set去重)。
    4. 循环取出算子，调用算子的backward方法，计算本层输入的局部梯度；
       将上游梯度与局部梯度链式相乘，把梯度累积到各个输入tensor的grad成员。

    注意：
        - 梯度为原地累积模式，多次调用backward会梯度累加；
        - 计算图遍历使用显式栈循环实现，不使用递归，规避递归深度溢出；
        - seen_set标记已经入队的算子，防止计算图中共享子图造成重复入队；
        - 叶子Tensor的grad保存最终结果，非叶子tensor仅做中间梯度传递。
    """
    if tensor.creator is None:
        return

    # TODO:梯度不允许被修改，不用init，应该是只读静态变量
    if tensor.grad is None:
        tensor.init_grad()

    creators: list[IOperator] = []
    seen_set: set[IOperator] = set()

    def add_creator(creator: IOperator):
        if creator not in seen_set:
            seen_set.add(creator)
            creators.append(creator)
            creators.sort(key=lambda x: x.generation)

    add_creator(tensor.creator)

    while creators:
        creator = creators.pop()
        gys: list[DataArray] = [output().grad for output in creator.outputs]
        gxs = creator.backward(*gys)
        if not isinstance(gxs, tuple):
            gxs = (gxs,)
        for x, gx in zip(creator.inputs, gxs):
            if x.grad is None:
                x.grad = gx
            else:
                x.grad = x.grad + gx
            if x.creator is not None:
                add_creator(x.creator)


def numerical_diff(
    func,
    inputs: tuple[Any],
    params: list[DataArray] = None,
    proj_v: Optional[np.ndarray] = np.array(1),
) -> list[DataArray]:
    r"""
    有限差分(中心差分)实现数值微分，用于梯度检查。

    func支持输出张量：提供proj_v做向量投影构造标量目标
    $$g = \mathtt{sum}(Y \odot \mathrm{proj\_v})$$
    将多维张量输出压缩为标量。

    对params内每一个参数逐元素做正负双向微小扰动，使用二阶精度中心差分近似偏导数。
    公式：
    .. math::
        \frac{\partial f}{\partial x_i} \approx \frac{f(x+h e_i)-f(x-h e_i)}{2h}

    本函数仅调用函数前向求值，**完全不依赖自动微分计算图**；
    专门用来校验 autograd 反向传播(VJP)实现的正确性，**不可用于模型训练**。

    Notes:
        1. 自适应扰动步长：$h = \sqrt{\epsilon_{\text{float64}}} \cdot (1+|x|)$，
        平衡截断误差与浮点数相消误差，epsilon取float64机器精度约2.22e‑16。
        2. 算法复杂度正比于params全部参数的总元素数量。
        **仅适用于小规模参数做梯度检查，参数量大时运行会极慢**。
        3. 只扰动`params`中的张量；`inputs`作为func的固定实参全程保持不变，不参与求导。
        4. 原地修改参数元素完成扰动；计算结束后严格恢复原始数值。
        增加异常兜底逻辑，保证即使中途抛出异常，参数也不会被篡改。
        5. 默认行为：proj_v使用签名默认值时，内部自动生成与func输出同shape全1数组，等价模拟 `sum(Y).backward()`。

    Args:
        func: 待求导目标函数，调用签名 func(*inputs)。
            - proj_v = None：func直接返回标量，跳过投影；
            - proj_v 传入ndarray：func返回多维张量，执行投影求和得到标量。
        inputs: tuple，传递给func的固定位置实参；这部分变量不会被扰动。
        params: List[DataArray]，待求导、会被逐元素扰动的参数列表。
            如果 params=None，则 inputs 即为待扰动求导的对象（此时inputs必须是DataArray）,
            此处以微小误差进行中值微分，要求此处扰动参数必须是高精度浮点数。
        proj_v: 投影权重。
            - 默认值：内部自动生成与输出同shape全1数组；
            - 指定值：数组shape必须与func输出张量完全一致。
            模拟上游梯度矩阵，不指定时上游梯度全为1，和套了``sum``的效果一致。

    Returns:
        List[DataArray]: 和params顺序一一对应，每个DataArray为对应参数的数值差分梯度，shape、dtype与原参数完全匹配。
    """
    eps = np.finfo(np.float64).eps
    h_scale = np.sqrt(eps)  # ~1.49e-8，中心差分推荐缩放系数
    grad_result: list[DataArray] = []

    def _eval(*args):
        return float(np.sum(func(*args) * proj_v))

    if params is None:
        inputs: list[DataArray] = inputs
        inputs = [np.astype(item, np.float64) for item in inputs]
        params = inputs

    for param in params:
        orig_param = param.copy()  # 保存原始完整副本，兜底复原
        grad = np.zeros_like(orig_param, dtype=np.float64)
        flat_param = param.reshape(-1)
        flat_grad = grad.reshape(-1)
        num_elem = flat_param.size

        for idx in range(num_elem):
            xi = float(flat_param[idx])
            h = h_scale * (1.0 + abs(xi))

            # +h扰动
            flat_param[idx] = xi + h
            f_plus = _eval(*inputs)

            # -h扰动
            flat_param[idx] = xi - h
            f_minus = _eval(*inputs)

            diff_val = (f_plus - f_minus) / (2.0 * h)
            flat_grad[idx] = diff_val

            flat_param[idx] = xi

        # 兜底强制恢复原始参数
        param[...] = orig_param[...]
        grad_result.append(grad)
    return grad_result
