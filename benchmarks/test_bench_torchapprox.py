import pytest
import torch
import torchvision.models as models
from conftest import input_sizes

import torchapprox.layers as tal
from torchapprox.utils import get_approx_modules, inplace_conversion


def set_bench_type(net, bench_type, lut):
    approx_modules = get_approx_modules(net)

    for _, m in approx_modules:
        if bench_type == "baseline":
            m.inference_mode = tal.InferenceMode.QUANTIZED
        elif bench_type == "lut":
            m.inference_mode = tal.InferenceMode.APPROXIMATE
            m.fast_model = None
            m.approx_op.lut = lut
        else:
            m.inference_mode = tal.InferenceMode.APPROXIMATE
            m.fast_model = bench_type


@pytest.fixture()
def bench_torchapprox(bench_architecture):
    model = inplace_conversion(bench_architecture)
    model.to("cuda")
    return model


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available, skipping networks benchmark",
)
@pytest.mark.parametrize("bench_type", ["baseline", "mul8s_1KVL", "mul8s_1KR3", "lut"])
def test_bench_torchapprox(benchmark, bench_torchapprox, bench_type, lut):
    net = bench_torchapprox
    set_bench_type(net, bench_type, lut)
    dummy_x = torch.rand(input_sizes["imagenet"], device=torch.device("cuda"))

    def benchmark_fn(x):
        y = net(x)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    with torch.no_grad():
        benchmark(benchmark_fn, dummy_x)
