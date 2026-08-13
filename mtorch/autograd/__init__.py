from mtorch.interfaces import ITensor, IOperator
from mtorch.autograd.functional import jacobian


def backward(tensor: ITensor) -> None:
    if tensor.creator is None:
        return

    if tensor is None:
        tensor.init_grad()

    creators: list[IOperator] = []
    seen_set: set = set()

    def add_creator(creator):
        if creator not in seen_set:
            seen_set.add(creator)
            creators.append(creator)
            creators.sort(key=lambda x: x.generation)

    add_creator(tensor.creator)

    while creators:
        creator = creators.pop()
        gys = [output().grad for output in creator.outputs]
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


__all__ = ["backward", "jacobian"]
