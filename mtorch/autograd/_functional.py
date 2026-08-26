from mtorch._interfaces import ITensor, IOperator


def backward(tensor: ITensor) -> None:
    if tensor.creator is None:
        return

    if tensor.grad is None:
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


# TODO 多种微分、数据微分
def jacobian(func, inputs, create_graph=False, numercial=False):
    pass
