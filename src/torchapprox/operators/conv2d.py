# pylint: disable=missing-module-docstring, arguments-differ, abstract-method
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch

from .backend import approx, dwconv2d


@dataclass
class Conv2dArgs:
    """
    Container class to pass convolution parameters
    around in a convenient way
    """

    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]]
    padding: Union[int, Tuple[int, int]]
    dilation: Union[int, Tuple[int, int]]
    groups: int

    def backward_args(self) -> Dict[str, Any]:
        """
        Generate arguments required by backward pass
        for gradient calculation

        Returns:
            Dict populated with parameters required
            in backward pass
        """
        bwd_args = {
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }
        return bwd_args


def _conv_bwd_ste(grad, x, w, conf, require_input_grad=True, require_weight_grad=True):
    """
    Wrapper to reuse Conv2d gradient calculation
    for several approximate forward functions

    Args:
        grad: Upstream gradient
        x: Activations from forward pass
        w: Weight from forward pass
        conf: Conv2d parameters

    Returns:
        Gradients for activations and weights
    """
    grad_input = grad_weight = None
    if require_input_grad:
        grad_input = torch.nn.grad.conv2d_input(x.size(), w, grad, **conf)
    if require_weight_grad:
        grad_weight = torch.nn.grad.conv2d_weight(x, w.size(), grad, **conf)
    return grad_input, grad_weight


def _group_limits(group_idx: int, total_groups: int, channels: int) -> Tuple[int, int]:
    channels_per_group = channels // total_groups
    lower_idx = group_idx * channels_per_group
    upper_idx = (group_idx + 1) * channels_per_group
    return int(lower_idx), int(upper_idx)


def _affine_requantize(
    x_q, w_q, y_q, x_scale, x_zero_point, w_scale, w_zero_point, conv_args, out_dims
):
    y_q = y_q.view(y_q.size(0), y_q.size(1), -1)
    for group in range(conv_args.groups):
        in_ch_lower, in_ch_upper = _group_limits(
            group, conv_args.groups, conv_args.in_channels
        )
        out_ch_lower, out_ch_upper = _group_limits(
            group, conv_args.groups, conv_args.out_channels
        )
        x_unfold = torch.nn.functional.unfold(
            x_q[
                :,
                in_ch_lower:in_ch_upper,
                :,
            ],
            kernel_size=conv_args.kernel_size,
            padding=conv_args.padding,
            stride=conv_args.stride,
            dilation=conv_args.dilation,
        )
        kernels_flat = w_q[out_ch_lower:out_ch_upper].view(
            conv_args.out_channels // conv_args.groups, -1
        )
        y_q[:, out_ch_lower:out_ch_upper] += (
            kernels_flat.size(-1) * x_zero_point * w_zero_point
            - x_zero_point * kernels_flat.sum(axis=1).unsqueeze(0)[:, :, None].float()
            - w_zero_point * x_unfold.sum(axis=1).unsqueeze(1).float()
        )

    y_q *= x_scale * w_scale
    y_q = y_q.view(
        x_q.size(0),
        conv_args.out_channels,
        out_dims[0],
        out_dims[1],
    )
    return y_q


class ApproxConv2dOp(torch.autograd.Function):
    """
    Autograd wrapper around Im2Col/ApproxGeMM Conv2d operator
    """

    @staticmethod
    def forward(
        x,
        w,
        x_scale,
        x_zero_point,
        w_scale,
        w_zero_point,
        conv_args: Conv2dArgs,
        out_dims,
        lut,
    ):
        w_int = torch.round((w / w_scale) + w_zero_point).char()

        # Pre-allocate output tensor
        y_q = torch.empty(
            x.size(0),
            conv_args.out_channels,
            math.prod(out_dims),
            device=x.device,
            dtype=torch.int32,
        )

        for group in range(conv_args.groups):
            # Calculate lower and upper channel index for current group
            in_ch_lower, in_ch_upper = _group_limits(
                group, conv_args.groups, conv_args.in_channels
            )
            out_ch_lower, out_ch_upper = _group_limits(
                group, conv_args.groups, conv_args.out_channels
            )

            # Im2Col operation
            x_unfold = torch.nn.functional.unfold(
                x[
                    :,
                    in_ch_lower:in_ch_upper,
                    :,
                ],
                kernel_size=conv_args.kernel_size,
                padding=conv_args.padding,
                stride=conv_args.stride,
                dilation=conv_args.dilation,
            )
            x_unfold_int = torch.round((x_unfold / x_scale) + x_zero_point).char()

            # Reshape weights to 2D
            kernels_flat = w_int[out_ch_lower:out_ch_upper].view(
                conv_args.out_channels // conv_args.groups, -1
            )

            # ApproxGeMM
            y_q[:, out_ch_lower:out_ch_upper] = approx(
                kernels_flat,
                x_unfold_int,
                lut,
            )

        x_q = torch.round((x / x_scale) + x_zero_point)
        w_q = torch.round((w / w_scale) + w_zero_point)
        y_q = y_q.float()

        y_q = _affine_requantize(
            x_q,
            w_q,
            y_q,
            x_scale,
            x_zero_point,
            w_scale,
            w_zero_point,
            conv_args,
            out_dims,
        )

        # Reshape to correct output size
        return y_q

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        x, w, _, _, _, _, conv_args, _, _ = inputs
        ctx.save_for_backward(x, w)
        ctx.conf = conv_args.backward_args()

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        conf = ctx.conf
        grad_input, grad_weight = _conv_bwd_ste(
            grad, x, w, conf, ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        )
        return grad_input, grad_weight, None, None, None, None, None, None, None


class FastApproxConv2dOp(torch.autograd.Function):
    """
    torch.autograd.Function wrapper for High Througput model of ApproxConv2d operator
    uses fast model for forward pass and non-approximate gradients
    for backward pass (STE)
    """

    @staticmethod
    def forward(ctx, x, w, model, kwargs):
        ctx.save_for_backward(x, w)
        ctx.conf = kwargs
        y = model(torch.nn.functional.conv2d, x, w, kwargs)
        return torch.round(y)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        conf = ctx.conf
        grad_input, grad_weight = _conv_bwd_ste(grad, x, w, conf)
        return grad_input, grad_weight, None, None


class ApproxDWConv2dOp(torch.autograd.Function):
    """
    torch.autograd.Function wrapper for GPU-accelerated Depthwise Conv
    """

    @staticmethod
    def forward(ctx, x, w, lut, kwargs):
        ctx.save_for_backward(x, w)
        ctx.conf = kwargs

        x = x.char()
        w = w.char()

        res = dwconv2d(x, w, lut, kwargs["stride"], kwargs["padding"]).float()
        res.requires_grad_()
        return res

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        conf = ctx.conf
        grad_input, grad_weight = _conv_bwd_ste(grad, x, w, conf)
        return grad_input, grad_weight, None, None
