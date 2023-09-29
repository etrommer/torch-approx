import copy

import numpy as np
import pytest
import torch

from torchapprox import layers as tal
from torchapprox import utils
from torchapprox.operators.htp_models import (
    htp_models_mul8s,
    htp_models_mul12s,
    htp_models_mul16s,
)
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
    utils.wrap_quantizable(mn)
    quant.prepare_qat(mn, tal.layer_mapping_dict(), inplace=True)

    assert isinstance(mn.conv.wrapped, tal.ApproxConv2d)
    assert isinstance(mn.linear.wrapped, tal.ApproxLinear)

    approx_modules = utils.get_approx_modules(mn)
    assert len(approx_modules) == 2
    assert approx_modules[0][0] == "conv.wrapped"
    assert approx_modules[1][0] == "linear.wrapped"


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


weight_quant_configs = [
    quant.FakeQuantize.with_args(
        observer=quant.MinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=-128,
        quant_max=127,
    ),
    quant.FakeQuantize.with_args(
        observer=quant.MinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
        quant_min=-128,
        quant_max=127,
    ),
    quant.FakeQuantize.with_args(
        observer=quant.MovingAveragePerChannelMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_channel_affine,
        quant_min=-128,
        quant_max=127,
    ),
]

layer_configs = [
    (torch.nn.Linear, (4, 20), (20, 10), {}),
    (torch.nn.Conv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 1}),
    (torch.nn.Conv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 2}),
    (torch.nn.Conv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 4}),
    (torch.nn.Conv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 8}),
    (torch.nn.Conv2d, (2, 8, 4, 4), (8, 8, 3), {"groups": 8}),
    (torch.nn.Conv2d, (2, 16, 32, 32), (16, 16, 3), {"groups": 16}),
]

# We use only the _accurate_ HTP models for testing
# because they allow for comparing the output to the quantized
# implementation
htp_models = [
    htp_models_mul8s["accurate"],
    htp_models_mul12s["accurate"],
    htp_models_mul16s["accurate"],
]


def generate_models(layer_config, device, weights_qconfig=None):
    if weights_qconfig:
        default_qconfig = tal.ApproxLayer.default_qconfig()
        qconfig = quant.QConfig(
            activation=default_qconfig.activation, weight=weights_qconfig
        )
    else:
        qconfig = None

    layer_type, _, layer_args, layer_kwargs = layer_config
    layer = tal.ApproxWrapper(
        layer_type(*layer_args, **layer_kwargs, device=device), qconfig=qconfig
    )
    quant.prepare_qat(layer, tal.layer_mapping_dict(), inplace=True)
    layer.wrapped.inference_mode = tal.InferenceMode.APPROXIMATE
    ref_layer = copy.deepcopy(layer)
    ref_layer.wrapped.inference_mode = tal.InferenceMode.QUANTIZED

    return layer, ref_layer


@pytest.mark.parametrize("layer_config", layer_configs)
@pytest.mark.parametrize("weight_qconfig", weight_quant_configs)
def test_layer_fwd(device, layer_config, weight_qconfig):
    input_dims = layer_config[1]
    layer, ref_layer = generate_models(layer_config, device, weight_qconfig)

    x = torch.rand(input_dims, device=device)
    xref = copy.deepcopy(x)

    y = layer(x)
    yref = ref_layer(xref)

    assert torch.allclose(y, yref, atol=1e-7)


@pytest.mark.parametrize("htp_model", htp_models)
@pytest.mark.parametrize("layer_config", layer_configs)
def test_htp(device, layer_config, htp_model):
    input_dims = layer_config[1]
    layer, ref_layer = generate_models(layer_config, device)
    layer.wrapped.htp_model = htp_model

    x = torch.rand(input_dims, device=device)
    xref = copy.deepcopy(x)

    y = layer(x)
    yref = ref_layer(xref)

    assert torch.allclose(y, yref, atol=1e-7)


@pytest.mark.parametrize("layer_config", layer_configs)
@pytest.mark.parametrize("weight_qconfig", weight_quant_configs)
def test_layer_bwd(device, layer_config, weight_qconfig):
    input_dims = layer_config[1]
    layer, ref_layer = generate_models(layer_config, device, weight_qconfig)

    x1 = torch.rand(input_dims, device=device, requires_grad=True)
    x2 = copy.deepcopy(x1)
    ref_layer(x1).sum().backward()
    layer(x2).sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=5e-7)
    assert torch.allclose(
        ref_layer.wrapped.weight.grad, layer.wrapped.weight.grad, atol=5e-7
    )
    assert torch.allclose(
        ref_layer.wrapped.bias.grad, layer.wrapped.bias.grad, atol=5e-7
    )


@pytest.mark.parametrize("layer", layer_configs)
def test_layer_empty_lut(device, layer):
    approx_type, input_dims, layer_args, layer_kwargs = layer
    approx_type = tal.layer_mapping_dict()[approx_type]
    layer_kwargs["bias"] = False
    layer = approx_type(
        *layer_args,
        **layer_kwargs,
        device=device,
        qconfig=tal.ApproxLayer.default_qconfig()
    )

    layer.inference_mode = tal.InferenceMode.APPROXIMATE
    layer.approx_op.lut = np.zeros((256, 256))

    x = torch.randint(-128, 128, size=input_dims, device=device, dtype=torch.float32)
    res = layer(
        x, torch.tensor([1.0], device=device), torch.tensor([0.0], device=device)
    )

    assert torch.allclose(torch.zeros_like(res), res)


@pytest.mark.parametrize("layer", layer_configs)
def test_layer_noise(device, layer):
    approx_type, input_dims, layer_args, layer_kwargs = layer
    approx_type = tal.layer_mapping_dict()[approx_type]
    layer = approx_type(
        *layer_args,
        **layer_kwargs,
        device=device,
        qconfig=tal.ApproxLayer.default_qconfig()
    )
    layer.inference_mode = tal.InferenceMode.NOISE

    ref_layer = copy.deepcopy(layer)
    ref_layer.inference_mode = tal.InferenceMode.QUANTIZED

    x = torch.randint(-128, 128, size=input_dims, device=device, dtype=torch.float32)
    x_scale = torch.tensor([1.0], device=device)
    x_zero_point = torch.tensor([0.0], device=device)

    layer.stdev = 0.1
    assert not torch.allclose(
        layer(x, x_scale, x_zero_point), ref_layer(x, x_scale, x_zero_point)
    )
    layer.stdev = 0.0
    assert torch.allclose(
        layer(x, x_scale, x_zero_point), ref_layer(x, x_scale, x_zero_point)
    )
