# pylint: disable=missing-module-docstring, arguments-differ, abstract-method
import math

import torch

from torchapprox.operators.conv2d import (
    ApproxConv2dOp,
    ApproxDWConv2dOp,
    Conv2dArgs,
    FastApproxConv2dOp,
)

from .approx_layer import ApproxLayer


class ApproxConv2d(torch.nn.Conv2d, ApproxLayer):
    """
    Approximate 2D Convolution layer implementation
    """

    def __init__(self, *args, **kwargs):
        torch.nn.Conv2d.__init__(self, *args, **kwargs)
        ApproxLayer.__init__(self)
        self._opcount = None
        self.to(self.weight.device)

    @staticmethod
    def from_super(cls_instance: torch.nn.Conv2d):
        """
        Alias for from_conv2d
        """
        return ApproxConv2d.from_conv2d(cls_instance)

    @staticmethod
    def from_conv2d(conv2d: torch.nn.Conv2d):
        """
        Construct ApproxConv2d from torch.nn.Conv2d layer
        """
        has_bias = conv2d.bias is not None
        approx_instance = ApproxConv2d(
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size,
            stride=conv2d.stride,
            padding=conv2d.padding,
            dilation=conv2d.dilation,
            groups=conv2d.groups,
            bias=has_bias,
            padding_mode=conv2d.padding_mode,
        )

        with torch.no_grad():
            approx_instance.weight = conv2d.weight
            if has_bias:
                approx_instance.bias = conv2d.bias

        return approx_instance

    def use_fast_dwconv(self) -> bool:
        """
        Determine whether layer can be run using DWConv CUDA kernels

        Returns:
            - True if layer can be mapped to dwconv2d backend function
            - False otherwise
        """

        if not self.weight.is_cuda:
            return False
        if self.approx_op.lut is None:
            return False
        if self.dilation[0] > 1 or self.dilation[1] > 1:
            return False
        if self.groups != self.in_channels:
            return False
        if self.in_channels != self.out_channels:
            return False
        if self.padding_mode != "zeros":
            return False
        return True

    def output_dims(self, x):
        """
        Output width and height
        """

        def dim(idx):
            # Copied from
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            return math.floor(
                (
                    x.size(idx + 2)
                    + 2 * self.padding[idx]
                    - self.dilation[idx] * (self.kernel_size[idx] - 1)
                    - 1
                )
                / self.stride[idx]
                + 1
            )

        return (dim(0), dim(1))

    @property
    def opcount(self) -> int:
        if self._opcount is None:
            raise ValueError(
                "Conv layer Opcount not populated. Run forward pass first."
            )
        return self._opcount

    @property
    def fan_in(self) -> int:
        """
        Number of incoming connection for a single neuron
        """
        return self.in_channels * math.prod(self.kernel_size)

    @property
    def conv_args(self) -> Conv2dArgs:
        """
        Wrap layer configuration in dataclass for more convenient passing around
        """
        args = Conv2dArgs(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return args

    def baseline_fwd(self, x):
        return torch.nn.functional.conv2d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def quant_fwd(self, x):
        x_q = self.x_quantizer.fake_quant(x)
        w_q = self.w_quantizer.fake_quant(self.weight)
        y = torch.nn.functional.conv2d(
            x_q,
            w_q,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return y

    def approx_fwd(self, x):
        x_q = self.x_quantizer.quantize(x)
        w_q = self.w_quantizer.quantize(self.weight)

        if self.fast_model is not None:
            # Use HTP Model
            y = FastApproxConv2dOp.apply(
                x_q, w_q, self.fast_model, self.conv_args.backward_args()
            )
            y = torch.round(y)
        elif self.use_fast_dwconv():
            # Use accelerated DWConv kernels
            y = ApproxDWConv2dOp.apply(
                x_q, w_q, self.approx_op.lut, self.conv_args.backward_args()
            )
        else:
            # Use regular Im2Col/GeMM
            out_dims = self.output_dims(x)
            y = ApproxConv2dOp.apply(
                x_q,
                w_q,
                self.conv_args,
                out_dims,
                self.approx_op.lut,
            )

        # Dequantize
        y /= self.x_quantizer.scale_factor * self.w_quantizer.scale_factor
        return y

    # pylint: disable=arguments-renamed
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # calculate opcount for this layer using the input tensor size on first forward pass
        if self._opcount is None:
            self._opcount = int(
                math.prod(self.kernel_size)
                * math.prod(self.output_dims(x))
                * (self.in_channels / self.groups)
                * self.out_channels
            )

        # Reshape bias tensor to make it broadcastable
        bias = None if self.bias is None else self.bias[:, None, None]
        return ApproxLayer.forward(self, x, bias)
