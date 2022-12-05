import copy

import numpy as np
import pytest
import torch

import torchapprox.layers as tal
import torchapprox.utils.evoapprox as evoutil
from torchapprox.operators import htp_models as htp

try:
    import evoapproxlib
except ImportError:
    pytest.skip(
        "EvoApproxLib not found. Skipping dependent tests", allow_module_level=True
    )


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
