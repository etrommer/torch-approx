# TorchApprox Documentation
TorchApprox is a PyTorch extension that allows for using non-standard multiplication functions in PyTorch. This makes it easy to study the effects of [Approximate Hardware Multipliers](https://arxiv.org/pdf/2301.12181.pdf) on Neural Network tasks. Currently `torch.nn.Linear` and `torch.nn.Conv2d` subclasses are implemented. Approximate Multipliers can help lower the arithmetic footprint of NN hardware with only minor reductions in accuracy. By default, integer multipliers with up to 8x8-Bit inputs are supported. These bitwidths are the standard numerical format when deploying neural networks on constrained hardware.

The custom product function for the 8x8-Bit input space is expected to be pre-computed and saved as a Numpy array.

Quantization of intermediate results during training uses the [PyTorch Quantization API](https://pytorch.org/docs/stable/quantization.html). 

TorchApprox is a companion package to [agn-approx](https://github.com/etrommer/agn-approx). TorchApprox implements low-level primitives, operators and approximate layers, while agn-approx provides high-level neural network experimental setups.
