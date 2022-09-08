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

    def __init__(self, bitwidth: int = 8):
        ApproxQuantizer.__init__(self, bitwidth)
        self.alpha = torch.nn.Parameter(torch.tensor([2.2]), requires_grad=True)

    @property
    def scale_factor(self) -> float:
        return self.int_max / self.alpha.item()

    def fake_quant(self, x):
        self.int_max = self.int_max.to(x.device)
        return self.FakeQuant.apply(x, self.alpha, self.int_max)

    def quantize(self, x, rounded=True):
        self.int_max = self.int_max.to(x.device)
        return self.Quant.apply(x, self.alpha, self.int_max, rounded)

    class FakeQuant(torch.autograd.Function):
        """
        Differentiable Fake-Quantization Node
        """

        @staticmethod
        def forward(ctx, x, alpha, int_max):
            alpha = torch.abs(alpha)
            scale_factor = int_max / alpha
            ctx.save_for_backward(x, alpha)

            x_quant = torch.clamp(x, min=-alpha.item(), max=alpha.item())
            x_quant = torch.round(x_quant * scale_factor) / scale_factor
            return x_quant

        @staticmethod
        def backward(ctx, grad_x_quant):
            x, alpha = ctx.saved_tensors
            lower_bound = x < -alpha
            upper_bound = x > alpha
            x_range = ~(lower_bound | upper_bound)
            grad_alpha = torch.sum(grad_x_quant * (~x_range).float()).view(-1)
            return grad_x_quant * x_range.float(), grad_alpha, None, None

    # pylint: disable=duplicate-code
    class Quant(torch.autograd.Function):
        """
        Differentiable Integer Quantization Node
        """

        @staticmethod
        def forward(ctx, x, alpha, int_max, rounded):
            scale_factor = int_max / alpha.item()
            ctx.save_for_backward(scale_factor)
            x_quant = torch.clamp((scale_factor * x), min=-int_max, max=int_max)
            if rounded:
                x_quant = torch.round(x_quant)
            return x_quant

        @staticmethod
        def backward(ctx, grad_x_quant):
            (scale_factor,) = ctx.saved_tensors
            return grad_x_quant * scale_factor, None, None, None
