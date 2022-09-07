# pylint: disable=missing-module-docstring, arguments-differ, abstract-method
import torch

from .approx_layer import ApproxLayer
from .fast_models import fast_models


class ApproxLinear(torch.nn.Linear, ApproxLayer):
    """
    Approximate Linear Layer implementation
    """

    def __init__(self, *args, **kwargs):
        torch.nn.Linear.__init__(self, *args, **kwargs)
        ApproxLayer.__init__(self)
        self._opcount = torch.tensor(self.in_features * self.out_features).float()
        self.to(self.weight.device)

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

    def approx_fwd_fast(self, x):
        class FastModelLinear(torch.autograd.Function):
            """
            torch.autograd.Function wrapper for Fast model.
            uses fast model for forward pass and non-approximate gradients
            for backward pass (STE)
            """

            @staticmethod
            def forward(ctx, x: torch.Tensor, w: torch.Tensor, model: str):
                ctx.save_for_backward(x, w)
                return fast_models[model](torch.nn.functional.linear, x, w, {})

            @staticmethod
            def backward(ctx, grad):
                x, w = ctx.saved_tensors
                grad_input = torch.matmul(grad, w.T)
                grad_weight = torch.sum(torch.matmul(grad.transpose(1, 2), x), axis=0).T
                return grad_input, grad_weight, None

        return FastModelLinear.apply(x, self.weight, self.fast_model)

    # pylint: disable=arguments-renamed
    def forward(self, x):
        return ApproxLayer.forward(self, x, self.bias)
