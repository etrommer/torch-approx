# TorchApprox QuickStart

## Usage

Let's assume that you have a vanilla PyTorch CNN model on which we want to try approximate multiplications:
```python
import torchvision.models as models
model = models.mobilenet_v2()
```
First, we need to apply quantization to the model to have a meaningful conversion from floating-point weights and activations to integers. Integers are required since they are commonly used for deployment and are the input format for approximate multipliers. Due to an idiosyncracy in how PyTorch's quantization is implemented, we first need to wrap each layer that we want to quantize in a `torchapprox.layers.ApproxWrapper` instance:
```python
from torchapprox.utils import wrap_quantizable

# Wrap Linear and Conv2d layer instances
wrap_quantizable(model)
```
after that, the model can be converted using the regular `torch.ao.quantziation.prepare_qat` function.We supply a custom layer mapping to make sure that layers are replaced with TorchApprox' quantized layer implementations, rather than Pytorch's.
```python
import torch.ao.quantization as quant
import torchapprox.layers as tal

# Convert Linear and Conv2d layers to their quantized equivalents
quant.preprare_qat(model, tal.layer_mapping_dict(), inplace=True)
```
It is recommended to first run a few epochs of Quantization-aware training with accurate multiplications to calibrate weights and quantization parameters. This is done with a regular Pytorch training loop on the converted model. After the quantization parameters have been calibrated successfully, the model can be switched into approximate multiplication mode.
Additionally, we need to supply a Lookup Table of pre-computed approximate multiplication results.
The lookup table is a 2D Numpy array of size 256x256.
```python
import numpy as np
from torchapprox.utils import get_approx_modules

# We simply use the result of an accurate multiplication as an example.
# Adjust the contents of `lut` to suit your needs.
x = y = np.arange(256)
xx, yy = np.meshgrid(x, y)
lut = xx*yy

for _, m in get_approx_modules(model):
    m.inference_mode = tal.InferenceMode.APPROXIMATE
    m.approx_op.lut = lut
```
The next training loop will now implement multiplications `y = x * w` in all replaced layers as a lookup operation `y = lut[x][w]`.

The companion project [agn-approx](https://github.com/etrommer/agn-approx) wraps these primitives in a high-level API using [pytorch-lightning](https://lightning.ai) and can be used as a reference or starting point for a less verbose implemnetation of experiments.

## Lookup Table Ordering
For unsigned multipliers, both axis of the LUT need to be ordered numerically:
```
x = y = [0, 1, 2, ..., 254, 255]
```
For signed multipliers, the axes  of the lookup table are **not** ordered numerically, but by the numerical order of the _unsigned_ twos-complement equivalent of each index, i.e.:
```
x = y = [0, 1, 2, ... 126, 127, -128, -127, ...-2, -1]
```
