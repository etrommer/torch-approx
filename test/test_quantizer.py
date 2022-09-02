import copy

import pytest
import torch

from torchapprox.quantizers import PACTQuant


def test_pact_fake():
    x = torch.rand([20, 20])
    dut = PACTQuant()
    x_quant = dut.fake_quant(x)
    assert torch.lt(x_quant, dut.alpha.item()).all()
    assert torch.gt(x_quant, -dut.alpha.item()).all()
    assert len(torch.unique(x_quant)) <= 2**dut.bitwidth


def test_pact_quant():
    dut = PACTQuant()
    x1 = torch.rand([20, 20])
    x2 = copy.deepcopy(x1)

    x1_fakequant = dut.fake_quant(x1)
    x2_quant = dut.quantize(x2).float()
    assert torch.allclose(x1_fakequant, x2_quant / dut.scale_factor)
    assert torch.lt(x2_quant, dut.int_max).all()
    assert torch.gt(x2_quant, -dut.int_max).all()


def test_pact_grad():
    dut = PACTQuant()
    x1 = torch.rand([20, 20], requires_grad=True)
    x2 = copy.deepcopy(x1)

    dut.fake_quant(x1).sum().backward()
    torch.div(dut.quantize(x2), dut.scale_factor).sum().backward()
    assert torch.allclose(x1.grad, x2.grad)
