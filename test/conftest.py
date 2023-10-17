import os

import numpy as np
import pytest
import torch


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
    return torch.from_numpy(xx * yy).int()


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
    return request.param
