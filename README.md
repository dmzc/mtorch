# mtorch

An autograd & neural network framework implemented purely on NumPy.

## Overview

mtorch implements reverse-mode automatic differentiation, similar to PyTorch.
It is a minimal deep learning framework designed for learning how autograd
and neural networks work under the hood — every operator ships its own
`forward` and `backward` implementation, with no heavy dependencies beyond NumPy.

- `Tensor` with automatic differentiation and operator overloading (`+ - * / @ **` etc.)
- Common operators: `Add`, `Mul`, `Matmul`, `Pow`, `Sin`, `Cos`, `Tanh`, `Exp`, `Log`, `Sum`, `Reshape`, `Transpose`, `BroadcastTo` ...
- Activation functions: `Sigmoid`, `ReLU`, `Tanh`, `Softmax`, `LogSoftmax`
- Loss functions: `MeanSquareLoss`, `CrossEntropyLoss`
- Neural network modules: `Module`, `Sequential`, `Linear`
- Optimizers: `SGD`, `Adam`
- Utilities: `Dataset`, `DataLoader`, `SprialDataset`, `UnivariateFunctionDataset`, `render`, `jacobian`
- Pure NumPy backend, no extra heavy dependencies

## Installation

```bash
pip install mtorch
```

### Requirements

- Python >= 3.10
- NumPy >= 2.2.6

## Quick Start

A linear regression example that fits `y = sin(2πx) + 2` with a small MLP
(see [mtorch/examples/nn/linear_regression.py](mtorch/examples/nn/linear_regression.py)):

```python
import numpy as np
import matplotlib.pyplot as plt
from mtorch import (
    Sequential, Linear, Sigmoid, SGD, MeanSquareLoss,
    UnivariateFunctionDataset, DataLoader,
)

# 超参数
lr = 0.2
epoch = 10000
batch_size: int = None

model = Sequential(
    Linear(1, 10),
    Sigmoid(),
    Linear(10, 1),
)
optimzer = SGD(model, lr=lr)
losser = MeanSquareLoss()

dataloader = DataLoader(
    dataset=UnivariateFunctionDataset(
        func=lambda x: np.sin(np.pi * 2 * x) + 2, data_size=100, batch_size=batch_size
    ),
)
for index in range(epoch):
    for x, y in dataloader:
        y_pred = model.forward(x)
        loss = losser.forward(y_actual=y_pred, y_expect=y)
        model.clear_grads()
        loss.backward()
        optimzer.step()
        print(f"第{index}轮损失{loss.data}")
```

### Using tensors directly

```python
import mtorch
from mtorch import Tensor

x = Tensor([1.0, 2.0, 3.0])
y = (x ** 2).sum()      # forward
y.backward()            # backward
print(x.grad)           # gradient dy/dx
```

## Project Structure

```
mtorch/
├── mtorch/
│   ├── tensor.py            # Tensor with autograd
│   ├── operator.py          # autograd operators (forward/backward)
│   ├── autograd/
│   │   └── functional.py    # jacobian, etc.
│   ├── nn/
│   │   └── modules/         # Module, Sequential, Linear, activation, loss, softmax
│   ├── optim/               # SGD, Adam
│   ├── utils/
│   │   ├── data/            # Dataset, DataLoader
│   │   ├── evaluation/
│   │   ├── render/          # computation graph rendering
│   │   └── trainer/
│   ├── tests/
│   └── examples/
│       └── nn/linear_regression.py
├── pyproject.toml
└── LICENSE
```

## Development

### Setup

Clone the repo and install in editable mode with dev dependencies:

```bash
git clone <repo-url>
cd mtorch
pip install -e ".[dev]"
```

Dev dependencies (from [pyproject.toml](pyproject.toml)) include `pytest`, `build` and `twine`.

### Running Tests

Tests live in [mtorch/tests/](mtorch/tests) and cover operators, autograd, losses, and the data pipeline:

- [test_operators.py](mtorch/tests/test_operators.py) — `test_add`, `test_mul`, `test_matmul`, `test_sin`, `test_sum`, `test_broadcast_to`, `test_softmax`, `test_crossEntroyLoss`, ...
- [test_data.py](mtorch/tests/test_data.py) — `test_dataset`, `test_dataLoader`

Run the full suite:

```bash
pytest
```

Run a specific test with verbose output and a markdown report (matches the VSCode launch config in [.vscode/launch.json](.vscode/launch.json)):

```bash
pytest ./mtorch/tests/test_data.py::test_dataLoader -vv -s --rootdir=./mtorch --md=./mtorch/tests/test_report.md
```

### Running Examples

Train the linear regression demo and view the fitted curve:

```bash
python -m mtorch.examples.nn.linear_regression
```

### Building the Package

Build sdist and wheel (uses `hatchling` as the build backend; `draft.py` is excluded via `tool.hatch.build`):

```bash
python -m build
```

## License

[MIT](LICENSE)
