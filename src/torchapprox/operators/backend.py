# pylint: disable=abstract-method, arguments-differ
"""
TorchApprox Accelerated Backend Functions
"""
import logging
import os

import torch
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)


sources = ["kernels/cpu/ta_gemm_cpu.cpp"]
extra_cflags = ["-fopenmp", "-O3"]

if torch.cuda.is_available():
    sources += [
        "kernels/cuda/ta_backend.cpp",
        "kernels/cuda/ta_gemm_cuda.cu",
        "kernels/cuda/ta_dwconv.cu",
    ]
    extra_cflags += ["-DTA_CUDA_EXTENSION"]
else:
    logger.warning("No CUDA device detected. Running on CPU.")

dirname = os.path.dirname(__file__)
sources = [os.path.join(dirname, src) for src in sources]
ta_backend = load(
    name="torchapprox_backend",
    sources=sources,
    extra_cflags=extra_cflags,
)


def dwconv2d(
    x: torch.FloatTensor,
    w: torch.FloatTensor,
    lut: torch.ShortTensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.FloatTensor:
    """
    Approximate 2D Depthwise Convolution
    """
    x = x.byte()
    w = w.byte()

    assert x.device == w.device
    assert x.is_cuda
    assert (
        x.dtype == w.dtype == torch.int8
    ), "Input operands need to be 8-Bit signed Integer"
    assert lut.dtype == torch.int32, "LUT needs to be 32 bit signed Integer"

    def make_tuple(val):
        if not isinstance(val, tuple):
            return (val, val)
        return val

    stride = make_tuple(stride)
    padding = make_tuple(padding)

    lut = lut.to(x.device)
    small = ta_backend.use_dwconv2d_small(x, w, 1, 1, *stride, *padding)
    if small:
        out = ta_backend.dwconv2d_small(x, w, lut, 1, 1, *stride, *padding, True)
    else:
        out = ta_backend.dwconv2d(x, w, lut, 1, 1, *stride, *padding, *padding, True)
    return out.float()


def approx(
    a: torch.Tensor,
    b: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    """
    Input validation wrapper for Approximate GeMM

    Expected are two matrices `a` and `b`, one of which needs to be batched
    """
    assert a.device == b.device, "Input Operands are on different devices"

    # Check input number formats
    assert a.dtype == b.dtype, "Input Operands are of different types"
    assert (
        a.dtype == torch.int8 or a.dtype == torch.uint8
    ), "Input operands need to be 8 bit Integer"
    assert lut.dtype == torch.int32, "LUT needs to be 32 bit signed Integer"

    # Check matrix dimensions
    if len(a.size()) == 3:
        assert len(b.size()) == 2, "Second operand is of wrong dimension"
        assert a.size(2) == b.size(
            0
        ), f"Matrix Product inner dimension does not match: {a.size()} - {b.size()}"
        batch_dim, dim_1, dim_2 = a.size(0), a.size(1), b.size(1)
        b = torch.transpose(b, 0, 1)
    elif len(b.size()) == 3:
        assert len(a.size()) == 2, "First operand is of wrong dimension"
        assert a.size(1) == b.size(
            1
        ), f"Matrix Product inner dimension does not match: {b.size()} - {a.size()}"
        batch_dim, dim_1, dim_2 = b.size(0), a.size(0), b.size(2)
        b = torch.transpose(b, 1, 2)
    else:
        raise ValueError(f"Incompatible Dimensions: {a.size()} - {b.size()}")
    res = torch.empty((batch_dim, dim_1, dim_2), dtype=torch.int32, device=a.device)
    lut = lut.to(a.device)
    a = a.contiguous()
    b = b.contiguous()

    if a.is_cuda:
        ta_backend.matmul_cuda(a, b, lut, res)
    else:
        ta_backend.matmul_cpu(a, b, lut, res)
    return res
