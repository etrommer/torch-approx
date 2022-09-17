from os import device_encoding

import pytest
import torch
from conftest import BATCH_SIZE, CONV2_DIM, channels

import torchapprox.layers as tal

model = ["lut", "mul8s_1KV8", "mul8s_1KR3", "mul8s_1L2D", "mul8s_1KVL"]


def trace_net(net, x):
    net(x)
    return torch.jit.trace(net, x)


@pytest.mark.parametrize("model", model)
@pytest.mark.parametrize("channels", channels)
def test_bench_torchapprox_conv2d(benchmark, model, lut, channels):
    dummy_x = torch.rand(
        (BATCH_SIZE, channels, CONV2_DIM, CONV2_DIM), device=torch.device("cuda")
    )
    layer = tal.ApproxConv2d(
        channels, channels, 3, padding=1, device=torch.device("cuda")
    )
    layer.inference_mode = tal.InferenceMode.APPROXIMATE
    if model == "lut":
        layer.approx_op.lut = lut
    else:
        layer.fast_model = model

    def benchmark_fn(x):
        y = layer(x)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    with torch.no_grad():
        benchmark(benchmark_fn, dummy_x)

    del layer
    del dummy_x
    torch.cuda.empty_cache()
