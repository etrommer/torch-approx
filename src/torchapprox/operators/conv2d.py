import imp
import math
from dataclasses import dataclass
from re import I
from typing import Tuple, Union

import torch

from .backend import approx
from .fast_models import fast_models


@dataclass
class Conv2dArgs:
    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple[int, int]]
    stride: Union[int, Tuple[int, int]]
    padding: Union[int, Tuple[int, int]]
    dilation: Union[int, Tuple[int, int]]
    groups: int

    def backward_args(self):
        bwd_args = {
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }
        return bwd_args


class ApproxConv2dOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        w,
        conv_args: Conv2dArgs,
        out_dims,
        lut,
    ):
        ctx.save_for_backward(x, w)
        ctx.conf = conv_args.backward_args()

        # Pre-allocate output tensor
        y = torch.empty(
            x.size(0),
            conv_args.out_channels,
            math.prod(out_dims),
            device=x.device,
            dtype=torch.int32,
        )

        for group in range(conv_args.groups):
            # Calculate lower and upper channel index for current group
            def limits(group, channels):
                group_size = int(channels / conv_args.groups)
                lower = group * group_size
                upper = (group + 1) * group_size
                return int(lower), int(upper)

            in_ch_lower, in_ch_upper = limits(group, conv_args.in_channels)
            out_ch_lower, out_ch_upper = limits(group, conv_args.out_channels)

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

            # Reshape weights to 2D
            kernels_flat = w[out_ch_lower:out_ch_upper].view(
                int(conv_args.out_channels / conv_args.groups), -1
            )

            # ApproxGeMM
            if lut is None:
                y[:, out_ch_lower:out_ch_upper] = kernels_flat @ x_unfold
            else:
                y[:, out_ch_lower:out_ch_upper] = approx(
                    kernels_flat.char(),
                    x_unfold.char(),
                    lut,
                )

        # Reshape to correct output size
        y = y.view(
            x.size(0),
            conv_args.out_channels,
            out_dims[0],
            out_dims[1],
        )
        y = y.float()
        y.requires_grad_()
        return y

    @staticmethod
    def backward(ctx, grad):
        grad_input, grad_weight = conv_bwd_ste(ctx, grad)
        return grad_input, grad_weight, None, None, None


def conv_bwd_ste(ctx, grad):
    x, w = ctx.saved_tensors
    conf = ctx.conf
    grad_input = torch.nn.grad.conv2d_input(x.size(), w, grad, **conf)
    grad_weight = torch.nn.grad.conv2d_weight(x, w.size(), grad, **conf)
    return grad_input, grad_weight


class FastModelConv2d(torch.autograd.Function):
    """
    torch.autograd.Function wrapper for Fast model.
    uses fast model for forward pass and non-approximate gradients
    for backward pass (STE)
    """

    @staticmethod
    def forward(ctx, x, w, model, kwargs):
        ctx.save_for_backward(x, w)
        ctx.conf = kwargs
        return fast_models[model](torch.nn.functional.conv2d, x, w, kwargs)

    @staticmethod
    def backward(ctx, grad):
        grad_input, grad_weight = conv_bwd_ste(ctx, grad)
        return grad_input, grad_weight, None, None
