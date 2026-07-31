from mtorch.interfaces import IOptimizer, IModule


class SGD(IOptimizer):

    __lr: float

    def __init__(self, params_obj: IModule, lr=0.1):
        super().__init__(params_obj)
        self.__lr = lr

    def step(self):
        lr = self.__lr
        for param in self._params_obj.params():
            param.data -= lr * param.grad.data
