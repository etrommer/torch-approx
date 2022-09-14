# pylint: disable=abstract-method, arguments-differ
"""
TorchApprox Accelerated Backend Functions
"""
import logging
import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)


sources = ["../../cpu/ta_gemm_cpu.cpp"]
extra_cflags = ["-fopenmp"]

if torch.cuda.is_available():
    sources += ["../../cuda/ta_backend.cpp", "../../cuda/ta_gemm_cuda.cu"]
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


def approx(
    a: torch.Tensor,
    b: torch.Tensor,
    lut: torch.Tensor,
    res: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Input validation wrapper for Approximate GeMM

    Expected are two matrices `a` and `b`, one of which needs to be batched
    """

    assert a.device == b.device, "Input Operands are on different devices"

    # Check input number formats
    assert a.dtype == b.dtype, "Input Operands are of different types"
    assert a.dtype == torch.int8, "Input operands need to be 8 bit signed Integer"
    assert lut.dtype == torch.int16, "LUT needs to be 32 bit signed Integer"

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
    if res is None:
        # Pre-allocate
        res = torch.empty((batch_dim, dim_1, dim_2), dtype=torch.int32, device=a.device)
    else:
        # Validate user-supplied results tensor
        assert res.dtype == torch.int32, "Result needs to be int32"
        assert a.device == res.device, "Results tensor on wrong device"
        assert len(res.size()) == 3
        assert (
            res.size(0) == batch_dim and res.size(1) == dim_1 and res.size(2) == dim_2
        ), "Results tensor shape does not match"
        res = res.contiguous()

    lut = lut.to(a.device)
    a = a.contiguous()
    b = b.contiguous()

    if a.is_cuda:
        ta_backend.matmul_cuda(a, b, lut, res)
    else:
        ta_backend.matmul_cpu(a, b, lut, res)
    return res
