from mtorch._interfaces import ITransform


class Compose(ITransform):
    r"""
    串行执行变换，前一个变换的输出是下一个变换的输入
    """

    __transforms: list[ITransform]

    def __init__(self, transforms: list[ITransform]):
        self.__transforms = transforms

    def __repr__(self):
        name = "->".join([transform for transform in self.__transforms])
        return f"Composed Transforms:{name}"

    def __call__(self, x):
        for tf in self.__transforms:
            x = tf(x)
        return


class Branch(ITransform):
    r"""
    并行执行变换，一个输入多个输出，每个输出互不干扰
    """

    pass


class RandomApply(ITransform):
    r"""
    整组以概率开启 TODO：
    """

    pass


class RandomChoice(ITransform):
    r"""
    随机选一个执行 TODO：
    """

    pass


class RandomOrder(ITransform):
    r"""
    打乱顺序然后顺序执行 TODO:
    """

    pass


class Conditional(ITransform):
    pass
