import copy

import numpy as np
import pytest
import torch

import torchapprox as ta


def test_linear():
    l = torch.nn.Linear(20, 10, bias=False)
    al = ta.layers.ApproxLinear.from_linear(copy.deepcopy(l))
    x = torch.rand(4, 20)

    al.inference_mode = ta.layers.InferenceMode.APPROXIMATE
    al.fast_model = "mul8s_1KV8"

    assert torch.allclose(l(x), al(x))


def test_conv2d():
    l = torch.nn.Conv2d(4, 4, 3)
    al = ta.layers.ApproxConv2d.from_conv2d(copy.deepcopy(l))
    x = torch.rand(2, 4, 4, 4)

    al.inference_mode = ta.layers.InferenceMode.APPROXIMATE
    al.fast_model = "mul8s_1KV8"

    assert torch.allclose(l(x), al(x))
