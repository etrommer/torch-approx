import copy
from itertools import product

import pytest
import torch

from torchapprox import ApproxMM, approx

sizes = [1, 2, 23, 115]


@pytest.fixture(params=product(sizes, sizes, sizes))
def matrices(request, device):
    """Generate a range of test matrices

    Args:
        request: Matrix dimensions
        device: The device to generate matrices on

    Yields:
        Test matrices A and B
    """

    def make_problem(batch_dim, dim1, dim2, dim3, device, dtype):
        """Helper function that constructs a problem of a given size

        Returns multipliable matrices a and b of dimensions:

        - a: batch_dim x dim0 x dim2
        - b: dim1 x dim3

        Args:
            batch_dim: Matrix A batch dimension
            dim0: Matrix A 1st dimension
            dim1: Matrix A 2nd and Matrix B 1st dimension
            dim2: Matrix B 2nd dimension
            device: The device to allocate problem matrices on ['cpu' or 'cuda']
            dtype: data type of A and B

        Returns:
            _description_
        """
        a = torch.randint(
            -128, 128, size=(batch_dim, dim1, dim2), device=device, dtype=dtype
        )
        b = torch.randint(-128, 128, size=(dim2, dim3), device=device, dtype=dtype)
        return a, b

    dim1, dim2, dim3 = request.param
    return make_problem(2, dim1, dim2, dim3, device, torch.int8)


def test_mm(matrices, lut):
    """
    Test correct GeMM result
    """
    a, b = matrices
    lut = lut.to(a.device)
    ref = torch.matmul(a.float(), b.float()).int()

    # res = a * b
    res = approx(a, b, lut)
    res_T = approx(b.T, a.transpose(1, 2), lut).transpose(1, 2)
    assert torch.allclose(ref, res)
    assert torch.allclose(ref, res_T)

    # Check correctness when 2nd operand is batched
    # res.T = b.T * a.T
    res_prealloc = approx(a, b, lut, torch.empty_like(ref))
    res_prealloc_T = approx(
        b.T, a.transpose(1, 2), lut, torch.empty_like(ref.transpose(1, 2))
    ).transpose(1, 2)
    assert torch.allclose(ref, res_prealloc)
    assert torch.allclose(ref, res_prealloc_T)


def test_mm_grad(device, lut):
    lut = lut.to(device)
    a1 = torch.randint(
        -128,
        128,
        size=(1, 10, 10),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    b1 = torch.randint(
        -128, 128, size=(10, 10), device=device, dtype=torch.float32, requires_grad=True
    )
    a2 = copy.deepcopy(a1)
    b2 = copy.deepcopy(b1)

    ApproxMM.apply(a1, b1, lut).sum().backward()
    torch.matmul(a2, b2).sum().backward()

    assert torch.allclose(a1.grad, a2.grad)
    assert torch.allclose(b1.grad, b2.grad)


def test_indexing(device):
    """
    Tests whether indexing into LUT uses the first operand for major axis and second operand for the minor axis
    """
    lut = torch.zeros((256, 256), dtype=torch.int16)
    lut[127, 0] = 42
    lut[0, 127] = -23

    res = ApproxMM.apply(torch.tensor([[[127.0]]]), torch.tensor([[0.0]]), lut)
    assert torch.allclose(res, torch.tensor([[[42.0]]]))

    res = ApproxMM.apply(torch.tensor([[[0.0]]]), torch.tensor([[127.0]]), lut)
    assert torch.allclose(res, torch.tensor([[[-23.0]]]))
