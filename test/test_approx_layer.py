import copy

import numpy as np
import pytest
import torch

import torchapprox as ta


def test_instantiate():
    with pytest.raises(TypeError):
        al = ta.layers.ApproxLayer()


def test_linear_from_super(device):
    l = torch.nn.Linear(20, 10, device=device)
    al = ta.layers.ApproxLinear.from_super(copy.deepcopy(l))

    # Properties should be identical
    assert l.weight.device == al.weight.device
    assert l.in_features == al.in_features
    assert l.out_features == al.out_features

    # Weights and biases should be the same
    assert torch.allclose(l.weight, al.weight)
    assert torch.allclose(l.bias, al.bias)

    # Baseline forward pass should yield the same result
    x = torch.rand(4, 20, device=device)
    assert torch.allclose(l(x), al(x))


def test_conv2d_from_super(device):
    l = torch.nn.Conv2d(8, 16, 3, device=device)
    al = ta.layers.ApproxConv2d.from_super(copy.deepcopy(l))

    # Check properties
    assert l.weight.device == al.weight.device
    assert l.kernel_size == al.kernel_size
    assert l.in_channels == al.in_channels
    assert l.out_channels == al.out_channels
    assert l.stride == al.stride
    assert l.padding == al.padding
    assert l.dilation == al.dilation
    assert l.groups == al.groups

    # Check weights and biases
    assert torch.allclose(l.weight, al.weight)
    assert torch.allclose(l.bias, al.bias)

    x = torch.rand(2, 8, 4, 4, device=device)
    assert torch.allclose(l(x), al(x))


def test_linear_properties():
    al = ta.layers.ApproxLinear(10, 20, False)
    assert al.fan_in == 10
    assert al.opcount == 200


def test_conv2d_properties():
    in_channels = 8
    out_channels = 16
    kernel_size = 3

    al = ta.layers.ApproxConv2d(in_channels, out_channels, kernel_size)
    x = torch.rand((4, in_channels, 4, 4))
    _ = al(x)
    assert al.fan_in == in_channels * kernel_size**2

    input_size = 2 * 2  # 4x4px without padding
    assert al.opcount == in_channels * input_size * kernel_size**2 * out_channels


layer_configs = [
    (ta.layers.ApproxLinear, (4, 20), (20, 10), {}),
    (ta.layers.ApproxConv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 1}),
    (ta.layers.ApproxConv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 2}),
    (ta.layers.ApproxConv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 4}),
    (ta.layers.ApproxConv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 8}),
]


@pytest.mark.parametrize("layer", layer_configs)
def test_layer_fwd(lut, device, layer):
    approx_type, input_dims, layer_args, layer_kwargs = layer

    layer = approx_type(*layer_args, **layer_kwargs, device=device)
    layer.inference_mode = ta.layers.InferenceMode.APPROXIMATE
    layer.approx_op.lut = lut

    ref_layer = copy.deepcopy(layer)
    ref_layer.inference_mode = ta.layers.InferenceMode.QUANTIZED

    x = torch.rand(input_dims, device=device)

    assert torch.allclose(ref_layer(x), layer(x), atol=5e-8)

    layer.approx_op.lut = None
    assert torch.allclose(ref_layer(x), layer(x), atol=5e-8)


@pytest.mark.parametrize("layer", layer_configs)
def test_layer_bwd(lut, device, layer):
    approx_type, input_dims, layer_args, layer_kwargs = layer

    layer = approx_type(*layer_args, **layer_kwargs, device=device)
    layer.inference_mode = ta.layers.InferenceMode.APPROXIMATE
    layer.approx_op.lut = lut

    ref_layer = copy.deepcopy(layer)
    ref_layer.inference_mode = ta.layers.InferenceMode.QUANTIZED

    x1 = torch.rand(input_dims, device=device, requires_grad=True)
    x2 = copy.deepcopy(x1)
    ref_layer(x1).sum().backward()
    layer(x2).sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=2.5e-7)
    assert torch.allclose(ref_layer.weight.grad, layer.weight.grad, atol=2.5e-7)
    assert torch.allclose(ref_layer.bias.grad, layer.bias.grad, atol=2.5e-7)


@pytest.mark.parametrize("layer", layer_configs)
def test_layer_empty_lut(device, layer):
    approx_type, input_dims, layer_args, layer_kwargs = layer

    layer = approx_type(*layer_args, **layer_kwargs, bias=False, device=device)
    layer.inference_mode = ta.layers.InferenceMode.APPROXIMATE
    layer.approx_op.lut = np.zeros((256, 256))

    x = torch.rand(input_dims, device=device)

    res = layer(x)
    assert torch.allclose(torch.zeros_like(res), res)


@pytest.mark.parametrize("layer", layer_configs)
def test_layer_noise(device, layer):
    approx_type, input_dims, layer_args, layer_kwargs = layer
    layer = approx_type(*layer_args, **layer_kwargs, device=device)
    layer.inference_mode = ta.layers.InferenceMode.NOISE

    ref_layer = copy.deepcopy(layer)
    ref_layer.inference_mode = ta.layers.InferenceMode.QUANTIZED

    x = torch.rand(input_dims, device=device)

    layer.stdev = 0.1
    assert not torch.allclose(layer(x), ref_layer(x))
    layer.stdev = 0.0
    assert torch.allclose(layer(x), ref_layer(x))
