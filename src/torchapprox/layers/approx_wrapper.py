from typing import Dict, Optional, Union

import torch
import torch.ao.quantization as tq

from .approx_conv2d import ApproxConv2d
from .approx_layer import ApproxLayer
from .approx_linear import ApproxLinear


class ApproxWrapper(torch.nn.Module):
    """
    Wrapper for adding quant/dequant stubs to a linear layer in a model.

    PyTorch provides the option to wrap modules in quantizers automatically,
    however a custom module is necessary so that we can forward the activation
    quantization scale and zero point to the approximate layer in the forward function.

    The wrapped instance of `torch.nn.Module` is meant to be replaced with an instance of
    `torchapprox.layers.ApproxLayer` in a separate call to
    `torch.ao.quantization.prepare()` after it has been wrapped here.
    """

    def __init__(
        self,
        wrapped: Union[torch.nn.Linear, torch.nn.Conv2d],
        qconfig: Optional[tq.QConfig] = None,
    ):
        """
        Wrap a torch.nn.linear layer with quantization stubs

        Args:
            wrapped: the layer to be wrapped
            qconfig: Quantization configuration. Defaults to None.
        """
        torch.nn.Module.__init__(self)
        self.quant_stub = tq.QuantStub()
        self.dequant_stub = tq.DeQuantStub()

        assert isinstance(wrapped, torch.nn.Linear) or isinstance(
            wrapped, torch.nn.Conv2d
        ), f"Received unknown layer type for wrapping: {type(wrapped)}"
        self.wrapped = wrapped

        if not qconfig:
            self.qconfig = ApproxLayer.default_qconfig()

    def forward(self, x):
        x_q = self.quant_stub(x)
        x_scale = getattr(self.quant_stub.activation_post_process, "scale", None)
        x_zero_point = getattr(
            self.quant_stub.activation_post_process, "zero_point", None
        )
        y_q = self.wrapped(x_q, x_scale, x_zero_point)
        y = self.dequant_stub(y_q)
        return y


def layer_mapping_dict() -> Dict[torch.nn.Module, ApproxLayer]:
    return {torch.nn.Linear: ApproxLinear, torch.nn.Conv2d: ApproxConv2d}
