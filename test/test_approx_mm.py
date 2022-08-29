import pytest
import torch

from torchapprox import ApproxMM, approx


def test_mm(test_inputs, lut):
    """
    Test correct GeMM result
    """
    a, b = test_inputs
    lut = lut.to(a.device)
    ref = torch.matmul(a.float(), b.float()).int()

    res = approx(a, b, lut)
    res_T = approx(b.T, a.transpose(1, 2), lut).transpose(1, 2)

    assert torch.allclose(ref, res)
    assert torch.allclose(ref, res_T)


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
