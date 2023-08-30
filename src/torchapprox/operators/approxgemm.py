# pylint: disable=abstract-method, arguments-differ, missing-module-docstring
from typing import Any, Tuple, TYPE_CHECKING

import torch

from torchapprox.operators.backend import approx

if TYPE_CHECKING:
    from torchapprox.layers.approx_layer import QuantizationParameters


class ApproxGeMM(torch.autograd.Function):
    """
    `torch.autograd.Function` wrapper for Approximate Matrix Multiplication
    This creates a differentiable graph node for the ApproxGeMM that uses
    the Straight-Through-Estimator to derive valid gradients
    """

    @staticmethod
    def forward(  # type: ignore
        x: torch.Tensor,
        w: torch.Tensor,
        lut: torch.Tensor,
        quant_params: "QuantizationParameters",
    ) -> torch.Tensor:
        """
        Approximate forward operation
        """
        x_int = torch.round(
            (x / quant_params.x_scale) + quant_params.x_zero_point
        ).char()[:, None, :]
        w_int = (
            torch.round((w / quant_params.w_scale) + quant_params.w_zero_point).char().T
        )
        res_int = approx(x_int, w_int, lut)
        res = res_int.float()
        scaled = (
            x.size(-1) * quant_params.x_zero_point * quant_params.w_zero_point
            - quant_params.x_zero_point * w_int.float().sum(axis=0)
            - quant_params.w_zero_point * x_int.float().sum(axis=-1)[:, None]
            + res
        ) * (quant_params.x_scale * quant_params.w_scale)
        scaled.requires_grad_()
        return scaled.view(scaled.size(0), -1)

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        (
            x,
            w,
            _,
            _,
        ) = inputs
        ctx.save_for_backward(x, w)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Calculate backward pass based on accurate matrix product (Straight-Through-Estimator)
        """
        x, w = ctx.saved_tensors
        # if len(a.size()) == 3:
        # Batched matrix is a
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(grad_output, w)
        if ctx.needs_input_grad[1]:
            grad_w = torch.matmul(grad_output.T, x)
        # else:
        #     # Batched matrix is b
        #     grad_a = torch.sum(torch.matmul(grad, b.transpose(1, 2)), axis=0)
        # grad_b = torch.matmul(grad.transpose(1, 2), a).transpose(1, 2)

        return grad_x, grad_w, None, None, None, None, None
