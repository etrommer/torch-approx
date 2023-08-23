# pylint: disable=missing-module-docstring, arguments-differ, abstract-method
from typing import Optional

import torch
import torch.ao.quantization as tq
import torch.nn as nn
from torch.ao.nn.qat.modules.linear import Linear as QATLinear

from torchapprox.operators.linear import FastLinearOp

from .approx_layer import ApproxLayer


class ApproxLinearWrapper(nn.Module):
    def __init__(self, wrapped: torch.nn.Linear, qconfig: Optional[tq.QConfig] = None):
        torch.nn.Module.__init__(self)
        self.quant_stub = tq.QuantStub()
        self.dequant_stub = tq.DeQuantStub()
        self.wrapped = wrapped
        if not qconfig:
            self.qconfig = ApproxLayer.default_qconfig()

    def forward(self, x):
        x_scale = getattr(self.quant_stub.activation_post_process, "scale", None)
        x_zero_point = getattr(
            self.quant_stub.activation_post_process, "zero_point", None
        )
        x_q = self.quant_stub(x)
        y_q = self.wrapped(x_q, x_scale, x_zero_point)
        y = self.dequant_stub(y_q)
        return y


class ApproxLinear(ApproxLayer, QATLinear):
    """
    Approximate Linear Layer implementation
    """

    _FLOAT_MODULE = nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        qconfig=None,
        device=None,
        dtype=None,
    ):
        QATLinear.__init__(
            self, in_features, out_features, bias, qconfig, device, dtype
        )
        ApproxLayer.__init__(self)
        self._opcount = torch.tensor(self.in_features * self.out_features).float()

    @property
    def fan_in(self) -> int:
        return int(self.in_features)

    @property
    def opcount(self) -> int:
        return int(self._opcount)

    def quant_fwd(self, x_q, w_q):
        y = torch.nn.functional.linear(x_q, w_q)
        return y

    def approx_fwd(self, x_q, w_q, x_scale, x_zero_point, w_scale, w_zero_point):
        if self.fast_model is None:
            # ApproxGeMM
            y = self.approx_op(x_q, w_q, x_scale, x_zero_point, w_scale, w_zero_point)
        else:
            # HTP Model
            y = FastLinearOp.apply(x_q, w_q, self.fast_model)
        # Rescale results
        # y /= x_scale * w_scale

        # y = y.view(y.size(0), -1)
        return y
