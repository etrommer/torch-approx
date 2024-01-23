# pylint: disable=missing-module-docstring, arguments-differ, abstract-method
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from torchapprox.layers.approx_layer import QuantizationParameters, TracedGeMMInputs

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

    def use_fast_dwconv(self) -> bool:
        """
        Determine whether layer can be run using DWConv CUDA kernels

        Returns:
            - True if layer can be mapped to dwconv2d backend function
            - False otherwise
        """

        if self.dilation[0] > 1 or self.dilation[1] > 1:
            return False
        if self.groups != self.in_channels:
            return False
        if self.in_channels != self.out_channels:
            return False
        return True


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


def _symmetric_requantize(y_q, quant_params):
    return y_q * quant_params.x_scale * quant_params.w_scale


def _affine_requantize(x_q, w_q, y_q, quant_params, conv_args, out_dims):
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
        # per-channel quantization only for weigths,
        # so correction factor for activations is the same for both modes
        y_q[:, out_ch_lower:out_ch_upper] += (
            -quant_params.x_zero_point
            * kernels_flat.sum(axis=1).unsqueeze(0)[:, :, None].float()
        )
        if len(quant_params.w_zero_point) == 1:
            # per-tensor weight quantization
            y_q[:, out_ch_lower:out_ch_upper] += (
                kernels_flat.size(-1)
                * quant_params.x_zero_point
                * quant_params.w_zero_point
                - quant_params.w_zero_point * x_unfold.sum(axis=1).unsqueeze(1).float()
            )
        else:
            # per-channel weight quantization
            y_q[:, out_ch_lower:out_ch_upper] += (
                kernels_flat.size(-1)
                * quant_params.x_zero_point
                * quant_params.w_zero_point[None, out_ch_lower:out_ch_upper, None]
                - quant_params.w_zero_point[None, out_ch_lower:out_ch_upper, None]
                * x_unfold.sum(axis=1)
                .unsqueeze(1)
                .expand(y_q[:, out_ch_lower:out_ch_upper].size())
                .float()
            )

    y_q *= quant_params.x_scale * quant_params.w_scale[None, :, None]
    return y_q


def _im2col_conv2d(
    x_q: torch.FloatTensor,
    w_q: torch.FloatTensor,
    conv_args: Conv2dArgs,
    lut: torch.ShortTensor,
    out_dims: Tuple[int, int],
    traced_inputs: Optional["TracedGeMMInputs"],
) -> torch.FloatTensor:
    # Pre-allocate output tensor
    y_q = torch.empty(
        x_q.size(0),
        conv_args.out_channels,
        math.prod(out_dims),
        device=x_q.device,
        dtype=torch.int32,
    )

    w_s8 = w_q.char()
    for group in range(conv_args.groups):
        # Calculate lower and upper channel index for current group
        in_ch_lower, in_ch_upper = _group_limits(
            group, conv_args.groups, conv_args.in_channels
        )
        out_ch_lower, out_ch_upper = _group_limits(
            group, conv_args.groups, conv_args.out_channels
        )

        # Im2Col operation
        x_unfold_s8 = torch.nn.functional.unfold(
            x_q[
                :,
                in_ch_lower:in_ch_upper,
                :,
            ],
            kernel_size=conv_args.kernel_size,
            padding=conv_args.padding,
            stride=conv_args.stride,
            dilation=conv_args.dilation,
        ).char()

        # Reshape weights to 2D
        w_flat_s8 = w_s8[out_ch_lower:out_ch_upper].view(
            conv_args.out_channels // conv_args.groups, -1
        )

        if traced_inputs:
            assert conv_args.groups == 1, "Tracing of depthwise Conv2D is not supported"
            traced_inputs.trace(x_unfold_s8, w_flat_s8)

        # ApproxGeMM
        y_q[:, out_ch_lower:out_ch_upper] = approx(
            w_flat_s8,
            x_unfold_s8,
            lut,
        )
    return y_q.float()


class ApproxConv2dOp(torch.autograd.Function):
    """
    Autograd wrapper around Im2Col/ApproxGeMM Conv2d operator
    """

    @staticmethod
    def forward(
        x: torch.FloatTensor,
        w: torch.FloatTensor,
        quant_params: "QuantizationParameters",
        conv_args: Conv2dArgs,
        htp_model: Optional[Callable],
        out_dims: Tuple[int, int],
        lut: torch.ShortTensor,
        traced_inputs: Optional["TracedGeMMInputs"],
    ):
        x_q = torch.round((x / quant_params.x_scale) + quant_params.x_zero_point)
        w_q = torch.round(
            (w / quant_params.w_scale[:, None, None, None])
            + quant_params.w_zero_point[:, None, None, None]
        )

        trace = traced_inputs is not None
        if htp_model is not None and not trace:
            # HTP model
            y_q = htp_model(
                torch.nn.functional.conv2d, x_q, w_q, conv_args.backward_args()
            )
            torch.round(y_q)
        elif (conv_args.use_fast_dwconv() and x.is_cuda and w.is_cuda) and not trace:
            # Depthwise Conv CUDA Kernel
            y_q = dwconv2d(x_q, w_q, lut, conv_args.stride, conv_args.padding)
        else:
            # im2col & gemm kernel (supports CPU & GPU)
            y_q = _im2col_conv2d(x_q, w_q, conv_args, lut, out_dims, traced_inputs)

        if quant_params.x_zero_point == 0 and torch.all(quant_params.w_zero_point == 0):
            y_q = _symmetric_requantize(y_q, quant_params)
        else:
            y_q = _affine_requantize(
                x_q,
                w_q,
                y_q,
                quant_params,
                conv_args,
                out_dims,
            )

        y_q = y_q.view(
            x_q.size(0),
            conv_args.out_channels,
            out_dims[0],
            out_dims[1],
        )

        return y_q

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        x, w, _, conv_args, _, _, _, _ = inputs
        ctx.save_for_backward(x, w)
        ctx.conf = conv_args.backward_args()

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        conf = ctx.conf
        grad_input, grad_weight = _conv_bwd_ste(
            grad, x, w, conf, ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        )
        return grad_input, grad_weight, None, None, None, None, None, None, None, None
