# pylint: disable=abstract-method, arguments-differ
"""
Implementation of quantization and fake-quantization using PACT algorithm
as described in https://arxiv.org/abs/1805.06085

This implementation differs by making the quantization symmetric around zero.

Implementation is partially derived from: https://github.com/KwangHoonAn/PACT/blob/master/module.py

Only the fake-quantization node updates the internal alpha paramter while the quantization node is static.
"""
import torch

from .approx_quantizer import ApproxQuantizer


class PACTQuant(ApproxQuantizer):
    """
    ApproxQuantizer implementation container for PACT algorithm
    """

    def __init__(self, bitwidth: int = 12):
        ApproxQuantizer.__init__(self, bitwidth)
        self.alpha = torch.nn.Parameter(torch.tensor([2.2]), requires_grad=True)

    @property
    def scale_factor(self) -> float:
        return self.int_max / self.alpha.item()

    def quantize(self, x):
        self.int_max = self.int_max.to(x.device)
        return self.Quant.apply(x, self.alpha, self.int_max)

    class Quant(torch.autograd.Function):
        """
        Differentiable Integer Quantization Node
        """

        @staticmethod
        def forward(ctx, x, alpha, int_max):
            scale_factor = int_max / alpha.item()
            ctx.save_for_backward(scale_factor)
            x_quant = torch.clamp((scale_factor * x), min=-int_max, max=int_max)
            x_quant = torch.round(x_quant)
            return x_quant

        @staticmethod
        def backward(ctx, grad_x_quant):
            (scale_factor,) = ctx.saved_tensors
            return grad_x_quant * scale_factor, None, None
