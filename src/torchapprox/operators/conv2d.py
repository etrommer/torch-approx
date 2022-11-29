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


def _conv_bwd_ste(grad, x, w, conf):
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
    grad_input = torch.nn.grad.conv2d_input(x.size(), w, grad, **conf)
    grad_weight = torch.nn.grad.conv2d_weight(x, w.size(), grad, **conf)
    return grad_input, grad_weight


class ApproxConv2dOp(torch.autograd.Function):
    """
    Autograd wrapper around Im2Col/ApproxGeMM Conv2d operator
    """

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
        x, w = ctx.saved_tensors
        conf = ctx.conf
        grad_input, grad_weight = _conv_bwd_ste(grad, x, w, conf)
        return grad_input, grad_weight, None, None, None


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
        return model(torch.nn.functional.conv2d, x, w, kwargs)

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
