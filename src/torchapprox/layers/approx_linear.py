# pylint: disable=missing-module-docstring
import torch

from .approx_layer import ApproxLayer


class ApproxLinear(torch.nn.Linear, ApproxLayer):
    """
    Approximate Linear Layer implementation
    """

    def __init__(self, *args, **kwargs):
        torch.nn.Linear.__init__(self, *args, **kwargs)
        ApproxLayer.__init__(self)
        self._opcount = torch.tensor(self.in_features * self.out_features).float()

    @staticmethod
    def from_super(cls_instance: torch.nn.Linear):
        """
        Alias for from_linear
        """
        return ApproxLinear.from_linear(cls_instance)

    @staticmethod
    def from_linear(linear: torch.nn.Linear):
        """
        Construct ApproxLinear from torch.nn.Linear layer
        """
        has_bias = linear.bias is not None
        approx_instance = ApproxLinear(
            linear.in_features, linear.out_features, bias=has_bias
        )

        with torch.no_grad():
            approx_instance.weight = linear.weight
            if has_bias:
                approx_instance.bias = linear.bias

        return approx_instance

    @property
    def fan_in(self) -> int:
        return int(self.in_features)

    @property
    def opcount(self) -> int:
        return int(self._opcount)

    def baseline_fwd(self, x):
        return torch.nn.functional.linear(x, self.weight)

    def quant_fwd(self, x):
        x_q = self.x_quantizer.fake_quant(x)
        w_q = self.w_quantizer.fake_quant(self.weight)
        y = torch.nn.functional.linear(x_q, w_q)
        return y

    def approx_fwd(self, x):

        # Quantize to Int8 range
        x_q = self.x_quantizer.quantize(x)[:, None, :]
        w_q = self.w_quantizer.quantize(self.weight)

        # ApproxGeMM
        y = self.approx_op(x_q, w_q.T)
        # Rescale results
        y /= self.x_quantizer.scale_factor * self.w_quantizer.scale_factor

        y = y.view(y.size(0), -1)
        return y

    # pylint: disable=arguments-renamed
    def forward(self, x):
        return ApproxLayer.forward(self, x, self.bias)
