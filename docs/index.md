# TorchApprox Documentation

## What is this?
TorchApprox is a PyTorch extension that allows for using non-standard multiplication functions for [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) in PyTorch. This makes it easy to study the effects of [Approximate Hardware Multipliers](https://arxiv.org/pdf/2301.12181.pdf) on Neural Network tasks. Approximate Multipliers are hardware multiplication units that don't guarantee mathematically exact results under all circumstances. When deployed on a hardware NN accelerator, Approximate Multipliers can
significantly lower the hardware resource consumption with only minimal impact on the Neural Network's accuarcy.

## Factsheet
- TorchApprox provides custom implementations of `torch.nn.Linear` and `torch.nn.Conv2d` layers.
- Integer multipliers with up to 8x8-Bit input operands are supported. 8-Bit integer quantization is the industry standard for deploying neural networks to edge devices, since it provides significantly smaller networks and compute requirements at almost no accuracy loss compared to FP16/FP32 models.
- The custom product function for the 8x8-Bit input space is expected to be pre-computed and provided as a 2D Numpy array. During runtime, multiplications in the supported layers types are replaced by array lookups into this table.
- TorchApprox supports quantization-aware and approximation-aware training. For quantization during training, it uses the [PyTorch Quantization API](https://pytorch.org/docs/stable/quantization.html). 
- TorchApprox is a companion package to [agn-approx](https://github.com/etrommer/agn-approx). TorchApprox implements low-level primitives, operators and approximate layers, while agn-approx provides high-level neural network experimental setups.

## Cite
If you use TorchApprox in your work, please cite it as:
```latex
@inproceedings{trommer23torchapprox,
  author       = {Elias Trommer and
                  Bernd Waschneck and
                  Akash Kumar},
  editor       = {Maksim Jenihhin and
                  Hana Kub{\'{a}}tov{\'{a}} and
                  Nele Metens and
                  Jaan Raik and
                  Foisal Ahmed and
                  Jan Belohoubek},
  title        = {High-Throughput Approximate Multiplication Models in PyTorch},
  booktitle    = {26th International Symposium on Design and Diagnostics of Electronic
                  Circuits and Systems, {DDECS} 2023, Tallinn, Estonia, May 3-5, 2023},
  pages        = {79--82},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/DDECS57882.2023.10139366},
  doi          = {10.1109/DDECS57882.2023.10139366},
}
```
