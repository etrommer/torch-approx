# pylint: disable=abstract-method, arguments-differ
"""
Implementation of quantization and fake-quantization using PACT algorithm
as described in https://arxiv.org/abs/1805.06085

This implementation differs by making the quantization symmetric around zero.

Implementation is partially derived from: https://github.com/KwangHoonAn/PACT/blob/master/module.py

"""
import torch

from .approx_quantizer import ApproxQuantizer


class FakeQuant(torch.autograd.Function):
    """
    Differentiable Fake-Quantization Node
    """

    @staticmethod
    def forward(ctx, x, alpha, int_max):
        alpha = torch.abs(alpha)
        y = torch.clamp(x, min=-alpha.item(), max=alpha.item())
        scale_factor = int_max / alpha
        y_quant = torch.round(y * scale_factor) / scale_factor
        ctx.save_for_backward(x, alpha)
        return y_quant

    @staticmethod
    def backward(ctx, grad_x_quant):
        x, alpha = ctx.saved_tensors
        lower_bound = x < -alpha
        upper_bound = x > alpha
        x_range = ~(lower_bound | upper_bound)
        grad_alpha = torch.sum(grad_x_quant * (~x_range).float()).view(-1)
        return grad_x_quant * x_range.float(), grad_alpha, None, None


class Quant(torch.autograd.Function):
    """
    Differentiable Integer Quantization Node
    """

    @staticmethod
    def forward(ctx, x, alpha, int_max):
        scale_factor = int_max / alpha.item()
        ctx.save_for_backward(torch.tensor([scale_factor]))
        x_quant = torch.clamp((scale_factor * x), min=-int_max, max=int_max)
        x_quant = torch.round(x_quant)
        return x_quant

    @staticmethod
    def backward(ctx, grad_x_quant):
        (scale_factor,) = ctx.saved_tensors
        return grad_x_quant * scale_factor, None, None


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
        return FakeQuant.apply(x, self.alpha, self.int_max)

    def quantize(self, x):
        return Quant.apply(x, self.alpha, self.int_max)
