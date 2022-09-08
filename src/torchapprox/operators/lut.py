# pylint: disable=missing-module-docstring
from typing import Optional, Union

import numpy as np
import torch

from .approxgemm import ApproxGeMM


class LUT(torch.nn.Module):
    """
    Class that wraps the Lookup Table matrix multiplication as a torch.nn.Module.
    This is required so that hooks can be attached in order to trace
    the quantized and unfolded input/output tensors at runtime
    """

    def __init__(self):
        torch.nn.Module.__init__(self)
        self._lut: Optional[torch.Tensor] = None

    @property
    def lut(self) -> Optional[torch.Tensor]:
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
    def lut(self, new_lut: Optional[Union[np.ndarray, torch.Tensor]]):
        if new_lut is None:
            self._lut = None
            return

        assert len(new_lut.shape) == 2, "LUT needs to be 2D square matrix"
        assert (
            new_lut.shape[0] == new_lut.shape[1] == 256
        ), "Only 8x8 Bit LUTs are currently supported."

        if isinstance(new_lut, torch.Tensor):
            assert new_lut.dtype == torch.short, "LUT needs to be signed 16 Bit Integer"
            self._lut = new_lut
        elif isinstance(new_lut, np.ndarray):
            self._lut = torch.from_numpy(new_lut).contiguous().short()
        else:
            raise ValueError(
                f"Unknown LUT input type: {type(new_lut)}, supported types: torch.Tensor, np.ndarray"
            )

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, res: Optional[torch.Tensor] = None
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
        if self.lut is None:
            # Ignore approximation LUT and output accurate matrix-matrix product.
            # Can be useful when piping data through this module is necessary but
            # approximate hardware simulation is not required
            # (e.g. for collecting activation maps in multipliers assignment stage)
            w = torch.round(w).float()
            x = torch.round(x).float()
            return x @ w
        return ApproxGeMM.apply(x, w, self.lut, res)
