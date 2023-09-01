import pytest
import torch

import torchapprox.layers as tal
from torchapprox.operators.htp_models import (
    htp_models_mul8s,
    htp_models_mul12s,
    htp_models_mul16s,
)
from torchapprox.utils.conversion import get_approx_modules, wrap_quantizable
from torch.ao.quantization import prepare_qat


def set_bench_type(net, bench_type, lut):
    approx_modules = get_approx_modules(net)

    for _, m in approx_modules:
        if bench_type == "lut":
            m.inference_mode = tal.InferenceMode.APPROXIMATE
            m.htp_model = None
            m.approx_op.lut = lut
        elif bench_type == "baseline":
            m.inference_mode = tal.InferenceMode.QUANTIZED
        else:
            m.inference_mode = tal.InferenceMode.APPROXIMATE
            m.htp_model = bench_type[1]


input_sizes = {
    "mnist": (128, 1, 28, 28),
    "cifar10": (128, 3, 32, 32),
    "imagenet": (16, 3, 224, 224),
}


@pytest.fixture()
def bench_torchapprox(bench_architecture):
    model = wrap_quantizable(bench_architecture)
    prepare_qat(model, tal.layer_mapping_dict(), inplace=True)
    model.eval()
    model.apply(torch.quantization.disable_observer)

    model.to("cuda")
    return model


base_models = ["baseline", "lut"]
fast_models = (
    list(htp_models_mul8s.items())
    + list(htp_models_mul12s.items())
    + list(htp_models_mul16s.items())
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available, skipping networks benchmark",
)
@pytest.mark.parametrize(
    "bench_type",
    base_models + fast_models,
    ids=base_models + [f[0] for f in fast_models],
)
def test_bench_torchapprox(benchmark, bench_torchapprox, bench_type, lut):
    net = bench_torchapprox
    set_bench_type(net, bench_type, lut)
    dummy_x = torch.rand(input_sizes["imagenet"], device=torch.device("cuda"))

    def benchmark_fn(x):
        torch.cuda.synchronize()
        net(x)
        torch.cuda.synchronize()

    with torch.no_grad():
        benchmark(benchmark_fn, dummy_x)
