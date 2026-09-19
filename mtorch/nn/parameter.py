from mtorch import Tensor


class Parameter(Tensor):

    @property
    def require_grad(self):
        return True

    @require_grad.setter
    def require_grad(self, value):
        raise RuntimeError("Parameter's require_grad always True!")

    def __repr__(self):
        data_str = str(self.data).replace("\n", "\n" + " " * 9)
        return f"Parameter( {data_str} )"
