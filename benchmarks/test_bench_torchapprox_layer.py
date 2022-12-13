from os import device_encoding

import pytest
import torch
from conftest import BATCH_SIZE, CONV2_DIM, channels

import torchapprox.layers as tal
from torchapprox.operators.htp_models import (
    htp_models_mul8s,
    htp_models_mul12s,
    htp_models_mul16s,
)


def trace_net(net, x):
    net(x)
    return torch.jit.trace(net, x)


base_models = ["lut"]
fast_models = (
    list(htp_models_mul8s.items())
    + list(htp_models_mul12s.items())
    + list(htp_models_mul16s.items())
)


@pytest.mark.parametrize(
    "bench_type",
    base_models + fast_models,
    ids=base_models + [f[0] for f in fast_models],
)
@pytest.mark.parametrize("channels", channels)
def test_bench_torchapprox_conv2d(benchmark, bench_type, lut, channels):
    torch.set_num_threads(24)
    dummy_x = torch.rand(
        (BATCH_SIZE, channels, CONV2_DIM, CONV2_DIM), device=torch.device("cuda")
    )
    layer = tal.ApproxConv2d(
        channels, channels, 3, padding=1, device=torch.device("cuda")
    )
    layer.inference_mode = tal.InferenceMode.APPROXIMATE

    if bench_type == "lut":
        layer.approx_op.lut = lut
    else:
        layer.fast_model = bench_type[1]

    def benchmark_fn(x):
        y = layer(x)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    with torch.no_grad():
        benchmark(benchmark_fn, dummy_x)

    del layer
    del dummy_x
    torch.cuda.empty_cache()
