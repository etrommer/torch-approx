# pylint: disable=missing-module-docstring, arguments-differ, abstract-method
import torch

from .fast_models import fast_models


class FastLinearOp(torch.autograd.Function):
    """
    torch.autograd.Function wrapper for Fast model.
    uses fast model for forward pass and non-approximate gradients
    for backward pass (STE)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, model: str):
        ctx.save_for_backward(x, w)
        return fast_models[model](torch.nn.functional.linear, x, w, {})

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        grad_input = torch.matmul(grad, w.T)
        grad_weight = torch.sum(torch.matmul(grad.transpose(1, 2), x), axis=0).T
        return grad_input, grad_weight, None
