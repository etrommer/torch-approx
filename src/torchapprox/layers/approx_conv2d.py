# pylint: disable=missing-module-docstring, arguments-differ, abstract-method
import math
from typing import Optional, Union

import torch
from torch.ao.nn.qat.modules.conv import Conv2d as QATConv2d

from torchapprox.operators.conv2d import (
    ApproxConv2dOp,
    Conv2dArgs,
)
from torch.nn.common_types import _size_2_t

from .approx_layer import ApproxLayer, QuantizationParameters


class ApproxConv2d(ApproxLayer, QATConv2d):
    """
    Approximate 2D Convolution layer implementation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        QATConv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            qconfig,
            device,
            dtype,
        )
        ApproxLayer.__init__(self)
        assert (
            padding_mode == "zeros"
        ), f"Unsupported padding_mode {padding_mode}, only zero-padding is supported"
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

    def quant_fwd(self, x_q, w_q):
        y = torch.nn.functional.conv2d(
            x_q,
            w_q,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return y

    def approx_fwd(self, x_q, w_q, quant_params: QuantizationParameters):
        y = ApproxConv2dOp.apply(
            x_q,
            w_q,
            quant_params,
            self.conv_args,
            self.fast_model,
            self.output_dims(x_q),
            self.approx_op.lut,
        )

        return y

    # pylint: disable=arguments-renamed
    def forward(
        self,
        x_q: torch.Tensor,
        x_scale: Optional[torch.Tensor] = None,
        x_zero_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # calculate opcount for this layer using the input tensor size on first forward pass
        if self._opcount is None:
            self._opcount = int(
                math.prod(self.kernel_size)
                * math.prod(self.output_dims(x_q))
                * (self.in_channels / self.groups)
                * self.out_channels
            )

        # Reshape bias tensor to make it broadcastable
        bias = None if self.bias is None else self.bias[:, None, None]
        return ApproxLayer.forward(self, x_q, x_scale, x_zero_point, bias)
