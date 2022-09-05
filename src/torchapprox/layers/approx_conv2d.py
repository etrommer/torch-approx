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

        out_dims = self.output_dims(x)
        y = torch.empty(
            x.size(0), self.out_channels, math.prod(out_dims), device=x.device
        )

        # in_group_size = int(self.in_channels / self.groups)
        # out_group_size = int(self.out_channels / self.groups)

        for group in range(self.groups):
            # Calculate lower and upper channel index for current group
            def limits(group, channels):
                group_size = int(channels / self.groups)
                lower = group * group_size
                upper = (group + 1) * group_size
                return int(lower), int(upper)

            in_ch_lower, in_ch_upper = limits(group, self.in_channels)
            out_ch_lower, out_ch_upper = limits(group, self.out_channels)

            x_unfold = torch.nn.functional.unfold(
                x_q[
                    :,
                    in_ch_lower:in_ch_upper,
                    :,
                ],
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
            )

            kernels_flat = w_q[out_ch_lower:out_ch_upper].view(
                int(self.out_channels / self.groups), -1
            )
            y[:, out_ch_lower:out_ch_upper] = self.approx_op(kernels_flat, x_unfold)

        y = y.view(x.size(0), self.out_channels, out_dims[0], out_dims[1])
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
