import math
import numpy as np
from mtorch.interfaces import IOptimizer, IModule


class Adam(IOptimizer):
    # TODO:目前只是模糊的知道公式，要详细了解下参数更新相关的细节
    def __init__(
        self, param_obj: IModule, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8
    ):
        super().__init__(params_obj=param_obj)
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    @property
    def lr(self):
        fix1 = 1.0 - math.pow(self.beta1, self.t)
        fix2 = 1.0 - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def step(self):
        self.t += 1
        for param in self._params_obj.params():
            key = id(param)
            if key not in self.ms:
                self.ms[key] = np.zeros_like(param.data)
                self.vs[key] = np.zeros_like(param.data)
            m, v = self.ms[key], self.vs[key]
            beta1, beta2, eps = self.beta1, self.beta2, self.eps
            grad = param.grad.data

            m += (1 - beta1) * (grad - m)
            v += (1 - beta2) * (grad * grad - v)
            param.data -= self.lr * m / (np.sqrt(v) + eps)
