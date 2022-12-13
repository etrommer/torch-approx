# pylint: disable=missing-module-docstring, arguments-differ, abstract-method
import torch


class FastLinearOp(torch.autograd.Function):
    """
    torch.autograd.Function wrapper for Fast model.
    uses fast model for forward pass and non-approximate gradients
    for backward pass (STE)
    """

    @staticmethod
    def forward(ctx, x, w, model):
        ctx.save_for_backward(x, w)
        return model(torch.nn.functional.linear, x, w, {})

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        grad_input = torch.matmul(grad, w)
        grad_weight = torch.matmul(grad.T, x)
        return grad_input, grad_weight, None
