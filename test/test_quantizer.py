import copy

import pytest
import torch

from torchapprox.quantizers import MinMaxQuant, PACTQuant


def quantizers():
    yield from [PACTQuant(), MinMaxQuant()]


@pytest.mark.parametrize("quantizer", quantizers())
def test_quant_fake(quantizer):
    dut = quantizer
    x = torch.rand([30, 30])
    x_quant = dut.fake_quant(x)
    assert len(torch.unique(x_quant)) <= 2**dut.bitwidth


@pytest.mark.parametrize("quantizer", quantizers())
def test_quant_fwd(quantizer):
    q1 = quantizer
    q2 = copy.deepcopy(q1)

    x1 = torch.rand([30, 30])
    x2 = copy.deepcopy(x1)

    x1_fakequant = q1.fake_quant(x1)
    x2_quant = q2.quantize(x2).float()

    assert torch.lt(x2_quant, q1.int_max + 1).all()
    assert torch.gt(x2_quant, -q1.int_max - 1).all()
    assert q1.scale_factor == q2.scale_factor
    assert torch.allclose(x1_fakequant, x2_quant / q2.scale_factor)


@pytest.mark.parametrize("quantizer", quantizers())
def test_quant_grad(quantizer):
    q1 = quantizer
    q2 = copy.deepcopy(q1)

    x1 = torch.rand([30, 30], requires_grad=True)
    x2 = copy.deepcopy(x1)

    q1.fake_quant(x1).sum().backward()
    torch.div(q2.quantize(x2), q2.scale_factor).sum().backward()

    assert torch.allclose(x1.grad, x2.grad)
