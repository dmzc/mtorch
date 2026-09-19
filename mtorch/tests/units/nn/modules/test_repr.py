from mtorch.nn.modules import Sequential, Linear, Relu


def test_repr():
    model = Sequential(Linear(100, 50), Relu(), Linear(50, 2))
    assert (
        repr(model)
        == "Sequential(\n\t\tLinear(input_size = 100, hidden_size = 50, use_bias = True), \n\t\tRelu(), \n\t\tLinear(input_size = 50, hidden_size = 2, use_bias = True)\n)"
    ), "模型repr"
