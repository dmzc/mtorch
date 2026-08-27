from mtorch._interfaces import ITensor, IOperator, DataArray
from typing import Any


def backward(tensor: ITensor) -> None:
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


# TODO 多种微分、数据微分
def jacobian(
    func: Any, inputs: Any, create_graph: bool = False, numercial: bool = False
):
    pass
