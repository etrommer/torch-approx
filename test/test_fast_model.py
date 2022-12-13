import copy

import numpy as np
import pytest
import torch

import torchapprox.layers as tal
from torchapprox.operators import htp_models as htp

try:
    import evoapproxlib

    import torchapprox.utils.evoapprox as evoutil
except ModuleNotFoundError:
    pytest.skip(
        "EvoApproxLib not found. Skipping dependent tests", allow_module_level=True
    )


layer_configs = [
    (tal.ApproxLinear, (4, 20), (20, 10), {}),
    (tal.ApproxConv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 1}),
    (tal.ApproxConv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 2}),
    (tal.ApproxConv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 4}),
    (tal.ApproxConv2d, (2, 8, 4, 4), (8, 16, 3), {"groups": 8}),
    (tal.ApproxConv2d, (2, 8, 4, 4), (8, 8, 3), {"groups": 8}),
]


@pytest.mark.parametrize("layer", layer_configs)
def test_layer_fwd(device, layer):
    approx_type, input_dims, layer_args, layer_kwargs = layer

    layer = approx_type(*layer_args, **layer_kwargs, device=device)
    layer.inference_mode = tal.InferenceMode.APPROXIMATE
    ref_layer = copy.deepcopy(layer)

    ref_layer.approx_op.lut = evoutil.lut("mul8s_1L12")
    layer.fast_model = htp.htp_models_mul8s["mul8s_1L12"]

    x = 2.0 * torch.rand(input_dims, device=device)

    assert torch.allclose(ref_layer(x), layer(x), atol=1e-7)


@pytest.mark.parametrize("layer", layer_configs)
def test_layer_bwd(device, layer):
    approx_type, input_dims, layer_args, layer_kwargs = layer

    layer = approx_type(*layer_args, **layer_kwargs, device=device)
    layer.inference_mode = tal.InferenceMode.APPROXIMATE
    ref_layer = copy.deepcopy(layer)

    ref_layer.approx_op.lut = evoutil.lut("mul8s_1L12")
    layer.fast_model = htp.htp_models_mul8s["mul8s_1L12"]

    x1 = torch.rand(input_dims, device=device, requires_grad=True)
    x2 = copy.deepcopy(x1)
    ref_layer(x1 * 2.0).sum().backward()
    layer(x2 * 2.0).sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=5e-7)
    assert torch.allclose(ref_layer.weight.grad, layer.weight.grad, atol=5e-7)
    assert torch.allclose(ref_layer.bias.grad, layer.bias.grad, atol=5e-7)


def test_linear():
    l = tal.ApproxLinear(20, 10, bias=False)
    l.inference_mode = tal.InferenceMode.APPROXIMATE
    al = copy.deepcopy(l)

    x = 2.0 * torch.rand(8, 20)

    l.approx_op.lut = evoutil.lut("mul8s_1L12")
    al.fast_model = htp.htp_models_mul8s["mul8s_1L12"]

    lres = l(x)
    alres = al(x)

    assert torch.allclose(l(x), al(x))


def test_conv2d():
    l = tal.ApproxConv2d(4, 4, 3, bias=False)
    l.inference_mode = tal.InferenceMode.APPROXIMATE
    al = copy.deepcopy(l)

    x = 2.0 * torch.rand(2, 4, 4, 4)

    # FIXME: LUT ordering seems inconsistent for reverse matmul kernel
    l.approx_op.lut = evoutil.lut("mul8s_1L12")
    al.fast_model = htp.htp_models_mul8s["mul8s_1L12"]

    lres = l(x)
    alres = al(x)

    assert torch.allclose(l(x), al(x))
