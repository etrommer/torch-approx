from os import device_encoding

import pytest
import torch

import torchapprox.layers as tal

channels = [1, 2, 4, 8, 16, 32, 64]
multipliers = ["mul8s_1KV8", "mul8s_1KR3", "mul8s_1L2D", "mul8s_1KVL"]


@pytest.mark.parametrize("multiplier", multipliers)
@pytest.mark.parametrize("channels", channels)
def test_bench_torchapprox_conv2d(benchmark, channels, multiplier):
    dummy_x = torch.rand((128, channels, 224, 224), device=torch.device("cuda"))
    layer = tal.ApproxConv2d(
        channels, channels, 3, padding=1, device=torch.device("cuda")
    )
    layer.fast_model = multiplier
    layer.inference_mode = tal.InferenceMode.APPROXIMATE

    def benchmark_fn(x):
        torch.cuda.synchronize()
        y = layer(x)
        torch.cuda.synchronize()

    benchmark(benchmark_fn, dummy_x)
    del layer
    del dummy_x
