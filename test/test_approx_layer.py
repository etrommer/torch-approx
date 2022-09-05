import copy

import numpy as np
import pytest
import torch

from torchapprox.layers import *


def layers():
    yield from [(ApproxLinear, (20, 10), (4, 20))]


def test_instantiate():
    with pytest.raises(TypeError):
        al = ApproxLayer()


@pytest.mark.parametrize("layer", layers())
def test_layer_from_super(device, layer):
    approx_type, layer_dims, input_dims = layer
    if approx_type == ApproxLinear:
        l = torch.nn.Linear(*layer_dims, device=device)
    al = approx_type.from_super(copy.deepcopy(l))

    # Properties should be identical
    assert l.weight.device == al.weight.device
    assert l.in_features == al.in_features
    assert l.out_features == al.out_features

    # Weights and biases should be the same
    assert torch.allclose(l.weight, al.weight)
    assert torch.allclose(l.bias, al.bias)

    # Baseline forward pass should yield the same result
    x = torch.rand(input_dims)
    assert torch.allclose(l(x), al(x))


@pytest.mark.parametrize("layer", layers())
def test_layer_fwd(lut, device, layer):
    approx_type, layer_dims, input_dims = layer

    layer = approx_type(*layer_dims, device=device)
    layer.inference_mode = InferenceMode.APPROXIMATE
    layer.approx_op.lut = lut

    ref_layer = copy.deepcopy(layer)
    ref_layer.inference_mode = InferenceMode.QUANTIZED

    x = torch.rand(input_dims, device=device)

    assert torch.allclose(ref_layer(x), layer(x))

    layer.approx_op.lut = None
    assert torch.allclose(ref_layer(x), layer(x))


@pytest.mark.parametrize("layer", layers())
def test_layer_empty_lut(device, layer):
    approx_type, layer_dims, input_dims = layer

    layer = approx_type(*layer_dims, bias=False, device=device)
    layer.inference_mode = InferenceMode.APPROXIMATE
    layer.approx_op.lut = np.zeros((256, 256))

    x = torch.rand(input_dims, device=device)

    res = layer(x)
    assert torch.allclose(torch.zeros_like(res), res)


@pytest.mark.parametrize("layer", layers())
def test_linear_noise(device, layer):
    approx_type, layer_dims, input_dims = layer
    layer = approx_type(*layer_dims, device=device)
    layer.inference_mode = InferenceMode.NOISE

    ref_layer = copy.deepcopy(layer)
    ref_layer.inference_mode = InferenceMode.QUANTIZED

    x = torch.rand(input_dims, device=device)

    layer.stdev = 0.1
    assert not torch.allclose(layer(x), ref_layer(x))
    layer.stdev = 0.0
    assert torch.allclose(layer(x), ref_layer(x))


def test_linear_properties():
    al = ApproxLinear(10, 20, False)
    assert al.fan_in == 10
    assert al.opcount == 200


def test_conv2d_properties():
    in_channels = 8
    out_channels = 16
    kernel_size = 3

    al = ApproxConv2d(in_channels, out_channels, kernel_size)
    x = torch.rand((4, in_channels, 4, 4))
    _ = al(x)
    assert al.fan_in == in_channels * kernel_size**2

    input_size = 2 * 2  # 4x4px without padding
    assert al.opcount == in_channels * input_size * kernel_size**2 * out_channels
