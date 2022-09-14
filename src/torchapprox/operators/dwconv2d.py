# pylint: disable=missing-module-docstring, arguments-differ, abstract-method
import torch

from torchapprox.operators.backend import dwconv2d


class ApproxDWConv2d(torch.autograd.Function):
    """
    torch.autograd.Function wrapper for GPU-accelerated Depthwise Conv
    """

    @staticmethod
    def forward(ctx, x, w, lut, kwargs):
        ctx.save_for_backward(x, w)
        ctx.conf = kwargs

        x = torch.round(x).char()
        w = torch.round(w).char()
        res = dwconv2d(x, w, lut, kwargs["stride"], kwargs["padding"]).float()
        res.requires_grad_()
        return res

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        conf = ctx.conf
        grad_input = torch.nn.grad.conv2d_input(x.size(), w, grad, **conf)
        grad_weight = torch.nn.grad.conv2d_weight(x, w.size(), grad, **conf)
        return grad_input, grad_weight, None, None
