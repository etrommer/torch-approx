# pylint: disable=abstract-method, arguments-differ
"""
Stateless Min/Max quantizer implementation
"""

import torch

from .approx_quantizer import ApproxQuantizer


class MinMaxQuant(ApproxQuantizer):
    """
    Min/Max-Quantizer container class
    """

    def __init__(self, bitwidth: int = 8):
        ApproxQuantizer.__init__(self, bitwidth)
        self._scale_factor: float = 0.0

    @property
    def scale_factor(self):
        return self._scale_factor

    def fake_quant(self, x):
        with torch.no_grad():
            minmax = torch.max(torch.abs(x))
            self._scale_factor = self.int_max / minmax
        return self.FakeQuant.apply(x, minmax, self.int_max)

    def quantize(self, x):
        with torch.no_grad():
            minmax = torch.max(torch.abs(x))
            self._scale_factor = self.int_max / minmax
        return self.Quant.apply(x, minmax, self.int_max)

    class FakeQuant(torch.autograd.Function):
        """
        Differentiable Fake-quantization node
        """

        @staticmethod
        def forward(ctx, x, minmax, int_max):
            scale_factor = int_max / minmax
            x_quant = torch.clamp(x, min=-minmax, max=minmax)
            x_quant = torch.round(x_quant * scale_factor) / scale_factor
            return x_quant

        @staticmethod
        def backward(ctx, grad_x_quant):
            return grad_x_quant, None, None

    class Quant(torch.autograd.Function):
        """
        Differentiable Integer Quantization node
        """

        @staticmethod
        def forward(ctx, x, minmax, int_max):
            scale_factor = int_max / minmax
            ctx.save_for_backward(torch.tensor([scale_factor]))
            x_quant = torch.clamp(x, min=-minmax, max=minmax)
            x_quant = torch.round(x_quant * scale_factor)
            return x_quant

        @staticmethod
        def backward(ctx, grad_x_quant):
            (scale_factor,) = ctx.saved_tensors
            return grad_x_quant * scale_factor, None, None
