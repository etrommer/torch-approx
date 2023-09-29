# pylint: disable=abstract-method, arguments-differ, missing-module-docstring
from typing import Any, Tuple, TYPE_CHECKING, Optional, Callable

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
        htp_model: Optional[Callable],
    ) -> torch.Tensor:
        """
        Approximate forward operation
        """

        x_q = torch.round((x / quant_params.x_scale) + quant_params.x_zero_point)[
            :, None, :
        ]
        w_q = torch.round(
            (w / quant_params.w_scale[:, None]) + quant_params.w_zero_point[:, None]
        ).T

        if htp_model is None:
            y_q = approx(x_q.char(), w_q.char(), lut).float()
        else:
            y_q = htp_model(torch.nn.functional.linear, x_q, w_q.T, {})

        if quant_params.x_zero_point != 0 or torch.any(quant_params.w_zero_point != 0):
            y_q = (
                x.size(-1) * quant_params.x_zero_point * quant_params.w_zero_point
                - quant_params.x_zero_point * w_q.float().sum(axis=0)
                - quant_params.w_zero_point
                * (x_q.float().sum(axis=-1).unsqueeze(-1).expand(y_q.size()))
                + y_q
            )
        y_q *= quant_params.x_scale * quant_params.w_scale
        return y_q.view(y_q.size(0), -1)

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        (
            x,
            w,
            _,
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
