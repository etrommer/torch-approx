import copy

import numpy as np
import pytest
import torch

from torchapprox.layers import ApproxLayer, ApproxLinear, InferenceMode


def test_instantiate():
    with pytest.raises(TypeError):
        al = ApproxLayer()


def test_linear_from_super(device):
    l = torch.nn.Linear(12, 8, device=device)
    al = ApproxLinear.from_super(l)
    assert l.weight.device == al.weight.device
    assert l.in_features == al.in_features
    assert l.out_features == al.out_features
    assert torch.allclose(l.weight, al.weight)
    assert torch.allclose(l.bias, al.bias)

    x = torch.rand((4, 12))
    assert torch.allclose(l(x), al(x))


def test_linear_properties():
    al = ApproxLinear(10, 20, False)
    assert al.fan_in == 10
    assert al.opcount == 200


def test_linear_fwd(lut, device):
    layer = ApproxLinear(20, 10, device=device)
    layer.inference_mode = InferenceMode.APPROXIMATE
    layer.approx_op.lut = lut

    ref_layer = copy.deepcopy(layer)
    ref_layer.inference_mode = InferenceMode.QUANTIZED

    x = torch.rand((4, 20), device=device)

    assert torch.allclose(ref_layer(x), layer(x))

    layer.approx_op.lut = None
    assert torch.allclose(ref_layer(x), layer(x))


def test_linear_empty_lut(device):
    layer = ApproxLinear(20, 10, bias=False, device=device)
    layer.inference_mode = InferenceMode.APPROXIMATE
    layer.approx_op.lut = np.zeros((256, 256))

    x = torch.rand((4, 20), device=device)

    res = layer(x)
    assert torch.allclose(torch.zeros_like(res), res)


def test_linear_noise(device):
    layer = ApproxLinear(20, 10, device=device)
    layer.inference_mode = InferenceMode.NOISE

    ref_layer = copy.deepcopy(layer)
    ref_layer.inference_mode = InferenceMode.QUANTIZED

    x = torch.rand((4, 20), device=device)

    layer.stdev = 0.1
    assert not torch.allclose(layer(x), ref_layer(x))
    layer.stdev = 0.0
    assert torch.allclose(layer(x), ref_layer(x))
