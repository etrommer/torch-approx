import copy

import pytest
import torch

import torchapprox.quantizers as taq

quantizers = [taq.PACTQuant(), taq.MinMaxQuant()]


@pytest.mark.parametrize("quantizer", quantizers)
def test_quant_fwd(quantizer, device):
    q = quantizer.to(device)

    x = 10 * torch.rand([30, 30], device=device)

    x_quant = q.quantize(x)

    assert torch.lt(x_quant, q.int_max + 1).all()
    assert torch.gt(x_quant, -q.int_max - 1).all()
    assert len(torch.unique(x_quant)) <= 2**q.bitwidth
