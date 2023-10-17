# pylint: disable=missing-module-docstring
from typing import Callable, Optional, Union, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

if TYPE_CHECKING:
    from torchapprox.layers.approx_layer import QuantizationParameters

from .approxgemm import ApproxGeMM


class LUTGeMM(torch.nn.Module):
    """
    Class that wraps the Lookup Table matrix multiplication as a torch.nn.Module.
    This is required so that hooks can be attached in order to trace
    the quantized and unfolded input/output tensors at runtime
    """

    def __init__(self):
        torch.nn.Module.__init__(self)
        self._lut: Optional[torch.Tensor] = None
        self.lut = self.accurate_lut()

    @staticmethod
    def accurate_lut() -> npt.NDArray[np.int32]:
        x = np.arange(256)
        x[x >= 128] -= 256
        xx, yy = np.meshgrid(x, x)
        return (xx * yy).astype(np.int32)

    @property
    def lut(self) -> torch.Tensor:
        """
        The Lookup table to use for approximate multiplication. LUT can be:
        - `None`: An accurate product is used internall. This is much faster than passing
            operands through LUT kernels. Functionally equivalent to running the layer in
            `quant` mode, but useful when the unfolded inputs/outputs need to be traced at runtime.
        - `torch.Tensor` or `numpy.array`:
            - 2D array of size 256x256 is required. Unused entries will be ignored when simulating
                multiplication where the operand width is less than 8 Bit
            - When supplying a `torch.Tensor` the datatype needs to be signed 16-Bit.
        """
        return self._lut

    @lut.setter
    def lut(self, new_lut: Union[np.ndarray, torch.Tensor]):
        assert len(new_lut.shape) == 2, "LUT needs to be 2D square matrix"
        assert (
            new_lut.shape[0] == new_lut.shape[1] == 256
        ), "Only 8x8 Bit LUTs are currently supported."

        if isinstance(new_lut, torch.Tensor):
            assert new_lut.dtype == torch.int, "LUT needs to be signed 32 Bit Integer"
            self._lut = new_lut
        elif isinstance(new_lut, np.ndarray):
            self._lut = torch.from_numpy(new_lut).contiguous().int()
        else:
            raise ValueError(
                f"Unknown LUT input type: {type(new_lut)}, supported types: torch.Tensor, np.ndarray"
            )

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        quant_params: "QuantizationParameters",
        htp_model: Optional[Callable],
    ) -> torch.Tensor:
        """
        Perform Approximate Matrix Multiply (GeMM)

        Args:
            x:       Activations Tensor of dimension B x N x M
            w:       Weight tensor of dimension K x M
            res:     Pre-allocated output tensor

        Returns:
            The approximate matrix batched matrix product of x and w, using the supplied LUT
        """
        return ApproxGeMM.apply(x, w, self.lut, quant_params, htp_model)
