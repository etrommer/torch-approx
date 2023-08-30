# pylint: disable=missing-module-docstring, arguments-differ, abstract-method

import torch
import torch.nn as nn
from torch.ao.nn.qat.modules.linear import Linear as QATLinear


from .approx_layer import ApproxLayer, QuantizationParameters


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

    def quant_fwd(self, x, w):
        return torch.nn.functional.linear(x, w)

    def approx_fwd(self, x, w, quant_params: QuantizationParameters):
        return self.approx_op(x, w, quant_params, self.htp_model)
