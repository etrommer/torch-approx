# pylint: disable=missing-module-docstring
import math

import torch

from .approx_layer import ApproxLayer


class ApproxConv2d(torch.nn.Conv2d, ApproxLayer):
    """
    Approximate 2D Convolution layer implementation
    """

    def __init__(self, *args, **kwargs):
        torch.nn.Conv2d.__init__(self, *args, **kwargs)
        ApproxLayer.__init__(self)
        self._opcount = None

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

    def baseline_fwd(self, x):
        return torch.nn.functional.conv2d(x, self.weight)

    def quant_fwd(self, x):
        pass

    def approx_fwd(self, x):
        pass

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
        bias = None if self.bias is None else self.bias[:, None, None]
        return ApproxLayer.forward(self, x, bias)
