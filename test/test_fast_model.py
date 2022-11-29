import copy

import numpy as np
import pytest
import torch

import torchapprox.layers as tal
from torchapprox.operators import htp_models as htp


def test_linear():
    l = torch.nn.Linear(20, 10, bias=False)
    al = tal.ApproxLinear.from_linear(copy.deepcopy(l))
    x = torch.rand(4, 20)

    al.inference_mode = tal.InferenceMode.APPROXIMATE
    al.fast_model = htp.htp_models_mul8s["accurate"]

    assert torch.allclose(l(x), al(x))


def test_conv2d():
    l = torch.nn.Conv2d(4, 4, 3)
    al = tal.ApproxConv2d.from_conv2d(copy.deepcopy(l))
    x = torch.rand(2, 4, 4, 4)

    al.inference_mode = tal.InferenceMode.APPROXIMATE
    al.fast_model = htp.htp_models_mul8s["accurate"]

    assert torch.allclose(l(x), al(x))
