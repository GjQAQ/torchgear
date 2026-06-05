Getting started
===============

Installation
------------

Install from source in editable mode::

   pip install -e .

PyTorch 2.0 or newer and Python 3.12+ are required.

Quick example
-------------

Tensor math helpers:

.. code-block:: python

   import torch
   from torchgear import abs2, expi

   z = torch.tensor([1 + 2j, 3 + 4j])
   abs2(z)

   x = torch.tensor([0.0, torch.pi / 2])
   expi(x)

Module base class with device/dtype helpers and freezing:

.. code-block:: python

   from torchgear.module import TorchgearModule
   import torch
   import torch.nn as nn

   class MyModel(TorchgearModule):
       def __init__(self):
           super().__init__()
           self.w = nn.Parameter(torch.randn(4))

       def forward(self, x):
           return x @ self.w

   model = MyModel()
   model.freeze()
   y = model.new_zeros(2)  # same device and dtype as parameters
