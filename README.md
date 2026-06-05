# torchgear

Utility extensions for PyTorch: tensor math helpers, shape utilities, and `nn.Module` mixins for device/dtype consistency and parameter freezing.

## Requirements

- Python 3.12+
- PyTorch 2.0+

## Installation

```bash
pip install torchgear
```

Install from source:

```bash
git clone https://github.com/GjQAQ/torchgear.git
cd torchgear
pip install -e .
```

## Quick start

```python
import torch
import torchgear as tg
import torch.nn as nn

# Math helpers
z = torch.tensor([1 + 2j, 3 + 4j])
tg.abs2(z)

# Module base class
class MyModel(tg.TorchgearModule):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4))

    def forward(self, x):
        return x @ self.w

model = MyModel()
model.freeze()
```

## Documentation

Online documentation: https://gjqaq.github.io/torchgear/

API reference and guides are also in the `docs/` directory. Build locally with [uv](https://docs.astral.sh/uv/) (install uv first):

```bash
uv sync --group doc
cd docs && make html
```

Without uv, install `sphinx` and `furo`, then run `make html` in `docs/`.

## Development

Requires [uv](https://docs.astral.sh/uv/) (see link above to install):

```bash
uv sync --group test
pytest
```

Without uv, install `pytest` and run `pytest` from the project root.

## License

MIT License. See [LICENSE](LICENSE) for details.
