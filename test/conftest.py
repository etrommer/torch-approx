import os
from itertools import product

import numpy as np
import pytest
import torch


def make_problem(batch_dim, dim1, dim2, dim3, device, dtype):
    """Helper function that constructs a problem of a given size

    Returns multipliable matrices a and b of dimensions:

    - a: batch_dim x dim1 x dim2
    - b: dim2 x dim3

    Args:
        batch_dim: Matrix A batch dimension
        dim1: Matrix A 1st dimension
        dim2: Matrix A 2nd and Matrix B 1st dimension
        dim3: Matrix B 2nd dimension
        device: The device to allocate problem matrices on ['cpu' or 'cuda']
        dtype: data type of A and B

    Returns:
        _description_
    """
    a = torch.randint(
        -128, 128, size=(batch_dim, dim1, dim2), device=device, dtype=dtype
    )
    b = torch.randint(-128, 128, size=(dim2, dim3), device=device, dtype=dtype)
    return (a, b)


@pytest.fixture
def lut():
    """Create accurate 8x8-Bit Lookup Table

    Returns:
        This populates a lookup table with accurate multiplication results
        for the operand range [-128, 127]
        LUT is created so that binary representation of
        the operand is monotically increasing when cast to an unsigned number, i.e.:
        `x = y = [0, 1,...,127, -128, -127,...-1]`
        This is done to simplify the conversion of the operand to an (unsigned)
        array index in the kernel later on.
    """
    x = np.arange(256)
    x[x >= 128] -= 256
    xx, yy = np.meshgrid(x, x)
    return torch.from_numpy(xx * yy).short()


@pytest.fixture(autouse=True)
def fix_seed():
    """
    Run before every test.
    - Fixes random seed to make test reproducible
    - Sets CUDA to blocking to allow for benchmarking of normally asynchronous kernels
    """
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    """
    Fixture to generate the available backends on the current device

    Args:
        request: Parameterized to either 'cpu' only for devices without CUDA
            or 'cpu' _and_ 'cuda' for devices with CUDA available

    Yields:
        - 'cpu' (always)
        - 'cuda' (if CUDA is available)
    """
    yield request.param


sizes = [1, 2, 7, 23, 115]


@pytest.fixture(params=product(sizes, sizes, sizes))
def test_inputs(request, device):
    """Generate a range of test matrices

    Args:
        request: Matrix dimensions
        device: The device to generate matrices on

    Yields:
        Test matrices A and B
    """
    dim1, dim2, dim3 = request.param
    yield make_problem(2, dim1, dim2, dim3, device, torch.int8)
