# pylint: disable=abstract-method, arguments-differ, missing-module-docstring
from typing import Optional

import torch

from torchapprox.operators.backend import approx


class ApproxGeMM(torch.autograd.Function):
    """
    `torch.autograd.Function` wrapper for Approximate Matrix Multiplication
    This creates a differentiable graph node for the ApproxGeMM that uses
    the Straight-Through-Estimator to derive valid gradients
    """

    @staticmethod
    def forward(  # type: ignore
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        lut: torch.Tensor,
        res: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Approximate forward operation
        """
        ctx.save_for_backward(a, b)
        a = torch.round(a).char()
        b = torch.round(b).char()
        res = approx(a, b, lut, res).float()
        res.requires_grad_()
        return res

    @staticmethod
    def backward(ctx, grad):
        """
        Calculate backward pass based on accurate matrix product (Straight-Through-Estimator)
        """
        a, b = ctx.saved_tensors
        if len(a.size()) == 3:
            # Batched matrix is a
            grad_a = torch.matmul(grad, b.T)
            grad_b = torch.sum(torch.matmul(grad.transpose(1, 2), a), axis=0).T
        else:
            # Batched matrix is b
            grad_a = torch.sum(torch.matmul(grad, b.transpose(1, 2)), axis=0)
            grad_b = torch.matmul(grad.transpose(1, 2), a).transpose(1, 2)

        return grad_a, grad_b, None, None
