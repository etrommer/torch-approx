import pytest
import torch
from adapt.approx_layers import axx_layers
from conftest import input_sizes
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn


def replace_modules(net):
    def replace_module(module):
        for n, m in module.named_children():
            for c in module.children():
                replace_module(c)
            if isinstance(m, torch.nn.Conv2d):
                has_bias = m.bias is not None
                ax_layer = axx_layers.AdaPT_Conv2d(
                    m.in_channels,
                    m.out_channels,
                    m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    groups=m.groups,
                    bias=has_bias,
                    padding_mode=m.padding_mode,
                )
                setattr(module, n, ax_layer)
            elif isinstance(m, torch.nn.Linear):
                has_bias = m.bias is not None
                ax_layer = axx_layers.AdaPT_Linear(
                    m.in_features, m.out_features, bias=has_bias
                )
                setattr(module, n, ax_layer)

    replace_module(net)


def collect_stats(model, data):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, image in enumerate([data]):
        model(image.cpu())

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(strict=False, **kwargs)
            print(f"{name:40}: {module}")


def calibrate(net, data):
    with torch.no_grad():
        stats = collect_stats(net, data)
        amax = compute_amax(net, method="percentile", percentile=99.99)


@pytest.fixture()
def bench_adapt(bench_architecture):
    threads = 24
    torch.set_num_threads(threads)

    model = bench_architecture
    model.to("cpu")
    replace_modules(model)
    dummy_x = torch.rand((128, 3, 224, 224), device=torch.device("cpu"))
    calibrate(model, dummy_x)
    return model


def test_bench_adapt(benchmark, bench_adapt):
    net = bench_adapt
    dummy_x = torch.rand(input_sizes["imagenet"], device=torch.device("cpu"))

    def benchmark_fn(x):
        y = net(x)

    benchmark(benchmark_fn, dummy_x)
