import copy

import numpy as np
import pytest
import torch

from torchapprox import layers as tal
from torchapprox import utils
import torch.ao.quantization as quant


def test_instantiate():
    with pytest.raises(TypeError):
        tal.ApproxLayer()


@pytest.mark.xfail
def test_compile(device, lut):
    layer = torch.nn.Linear(42, 23)
    w = tal.approx_linear.ApproxLinearWrapper(layer)
    x = torch.rand(128, 42).requires_grad_()
    quant.prepare_qat(w, {torch.nn.Linear: tal.ApproxLinear}, inplace=True)

    w.wrapped.approx_op.lut = lut
    w.wrapped.inference_mode = tal.InferenceMode.APPROXIMATE
    w_comp = torch.compile(w)
    w_comp(x)


def test_conversion():
    class MiniNet(torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.conv = torch.nn.Conv2d(3, 6, 3)
            self.linear = torch.nn.Linear(20, 10)

    mn = MiniNet()
    utils.inplace_conversion(mn)

    assert isinstance(mn.conv, tal.ApproxConv2d)
    assert isinstance(mn.linear, tal.ApproxLinear)

    approx_modules = utils.get_approx_modules(mn)
    assert len(approx_modules) == 2
    assert approx_modules[0][0] == "conv"
    assert approx_modules[1][0] == "linear"


def test_linear_from_super(device):
    lin_layer = torch.nn.Linear(20, 10, device=device)
    wrapped_lin = tal.ApproxWrapper(lin_layer)
    quant.prepare_qat(wrapped_lin, tal.layer_mapping_dict(), inplace=True)
    # al = tal.ApproxLinear.from_float(copy.deepcopy(l))

    # Properties should be identical
    assert lin_layer.weight.device == wrapped_lin.wrapped.weight.device
    assert lin_layer.in_features == wrapped_lin.wrapped.in_features
    assert lin_layer.out_features == wrapped_lin.wrapped.out_features

    # Weights and biases should be the same
    assert torch.allclose(lin_layer.weight, wrapped_lin.wrapped.weight)
    assert torch.allclose(lin_layer.bias, wrapped_lin.wrapped.bias)


def test_conv2d_from_super(device):
    conv_layer = torch.nn.Conv2d(8, 16, 3, device=device)
    wrapped_conv = tal.ApproxWrapper(conv_layer)
    quant.prepare_qat(wrapped_conv, tal.layer_mapping_dict(), inplace=True)

    # Check properties
    assert conv_layer.weight.device == wrapped_conv.wrapped.weight.device
    assert conv_layer.kernel_size == wrapped_conv.wrapped.kernel_size
    assert conv_layer.in_channels == wrapped_conv.wrapped.in_channels
    assert conv_layer.out_channels == wrapped_conv.wrapped.out_channels
    assert conv_layer.stride == wrapped_conv.wrapped.stride
    assert conv_layer.padding == wrapped_conv.wrapped.padding
    assert conv_layer.dilation == wrapped_conv.wrapped.dilation
    assert conv_layer.groups == wrapped_conv.wrapped.groups

    # Check weights and biases
    assert torch.allclose(conv_layer.weight, wrapped_conv.wrapped.weight)
    assert torch.allclose(conv_layer.bias, wrapped_conv.wrapped.bias)


def test_linear_properties():
    al = tal.ApproxLinear(10, 20, False)
    assert al.fan_in == 10
    assert al.opcount == 200


def test_conv2d_properties():
    in_channels = 8
    out_channels = 16
    kernel_size = 3

    al = tal.ApproxConv2d(in_channels, out_channels, kernel_size)
    x = torch.rand((4, in_channels, 4, 4))
    _ = al(x)
    assert al.fan_in == in_channels * kernel_size**2

    input_size = 2 * 2  # 4x4px without padding
    assert al.opcount == in_channels * input_size * kernel_size**2 * out_channels


layer_configs = [
    (tal.ApproxLinear, (4, 20), (20, 10), {}),
    (tal.ApproxConv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 1}),
    (tal.ApproxConv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 2}),
    (tal.ApproxConv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 4}),
    (tal.ApproxConv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 8}),
    (tal.ApproxConv2d, (2, 8, 4, 4), (8, 8, 3), {"groups": 8}),
    (tal.ApproxConv2d, (2, 16, 32, 32), (16, 16, 3), {"groups": 16}),
]


@pytest.mark.parametrize("layer", layer_configs)
def test_layer_fwd(lut, device, layer):
    approx_type, input_dims, layer_args, layer_kwargs = layer

    layer = approx_type(*layer_args, **layer_kwargs, device=device)
    layer.inference_mode = tal.InferenceMode.APPROXIMATE
    layer.approx_op.lut = lut

    ref_layer = copy.deepcopy(layer)
    ref_layer.inference_mode = tal.InferenceMode.QUANTIZED

    x = torch.rand(input_dims, device=device)

    assert torch.allclose(ref_layer(x), layer(x), atol=1e-7)

    layer.approx_op.lut = None
    assert torch.allclose(ref_layer(x), layer(x), atol=1e-7)


@pytest.mark.parametrize("layer", layer_configs)
def test_layer_bwd(lut, device, layer):
    approx_type, input_dims, layer_args, layer_kwargs = layer

    layer = approx_type(*layer_args, **layer_kwargs, device=device)
    layer.inference_mode = tal.InferenceMode.APPROXIMATE
    layer.approx_op.lut = lut

    ref_layer = copy.deepcopy(layer)
    ref_layer.inference_mode = tal.InferenceMode.QUANTIZED

    x1 = torch.rand(input_dims, device=device, requires_grad=True)
    x2 = copy.deepcopy(x1)
    ref_layer(x1).sum().backward()
    layer(x2).sum().backward()

    idxs = torch.nonzero(torch.abs(ref_layer.weight.grad - layer.weight.grad) > 1e-3)
    if len(idxs) != 0:
        print(idxs[0])
        print(ref_layer.weight.grad[0])
        print(layer.weight.grad[0])

    assert torch.allclose(x1.grad, x2.grad, atol=5e-7)
    assert torch.allclose(ref_layer.weight.grad, layer.weight.grad, atol=5e-7)
    assert torch.allclose(ref_layer.bias.grad, layer.bias.grad, atol=5e-7)


@pytest.mark.parametrize("layer", layer_configs)
def test_layer_empty_lut(device, layer):
    approx_type, input_dims, layer_args, layer_kwargs = layer

    layer = approx_type(*layer_args, **layer_kwargs, bias=False, device=device)
    layer.inference_mode = tal.InferenceMode.APPROXIMATE
    layer.approx_op.lut = np.zeros((256, 256))

    x = torch.rand(input_dims, device=device)

    res = layer(x)
    assert torch.allclose(torch.zeros_like(res), res)


@pytest.mark.parametrize("layer", layer_configs)
def test_layer_noise(device, layer):
    approx_type, input_dims, layer_args, layer_kwargs = layer
    layer = approx_type(*layer_args, **layer_kwargs, device=device)
    layer.inference_mode = tal.InferenceMode.NOISE

    ref_layer = copy.deepcopy(layer)
    ref_layer.inference_mode = tal.InferenceMode.QUANTIZED

    x = torch.rand(input_dims, device=device)

    layer.stdev = 0.1
    assert not torch.allclose(layer(x), ref_layer(x))
    layer.stdev = 0.0
    assert torch.allclose(layer(x), ref_layer(x))
