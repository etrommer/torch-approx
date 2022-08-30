# pylint: disable=missing-module-docstring
import logging
from abc import ABC, abstractmethod

import torch

from .approx_operator import LUT

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = [torch.nn.Linear, torch.nn.Conv2d]


class ApproxLayer(torch.nn.Module, ABC):
    """
    Derivable Abstract Base Class for implementing Approximate Neural Network layers
    """

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.alpha_x: torch.nn.Parameter = torch.nn.Parameter(
            torch.tensor(2.2), requires_grad=True
        )
        self.alpha_w: torch.Tensor = torch.Tensor([0.0])
        self.bitwidth: int = 8
        self.approx_op: LUT = LUT()

        self._stdev: torch.nn.Paramter = torch.nn.Parameter(
            torch.tensor(0.0), requires_grad=True
        )
        self._mode: str = "base"

    @property
    def mode(self) -> str:
        """
        Layer inference mode. Can be any of:
        - `base`: Run inference as unperturbed FP32 baseline
        - `quant`: Run inference using the layer's quantizer
        - `approx`: Run inference using approximate product LUT
        - `noise`: Run inference that is perturbed with additive Gaussian noise
        """
        return self._mode

    @mode.setter
    def mode(self, new_mode: str):
        supported_modes = ["base", "quant", "approx", "noise"]
        if not new_mode in supported_modes:
            raise ValueError(f"Trying to set unsupported mode: {new_mode}")
        self._mode = new_mode

    @property
    def int_max(self) -> int:
        """
        Maximum value for an operand according to currently set layer bitwidth.
        """
        return 2 ** (self.bitwidth - 1) - 1

    @property
    def stdev(self) -> torch.nn.Parameter:
        """
        The relative standard deviation of the Additive Gaussian noise that is added
        to the computation output. Scaling is done relative the current batch's standard devitaion.
        This is only used when the mode is set to `noise`. It will have no effect in other modes.
        """
        return self._stdev

    @stdev.setter
    def stdev(self, noise_std: float):
        self._stdev = torch.nn.Parameter(torch.tensor(noise_std), requires_grad=True)

    @abstractmethod
    def baseline_fwd(self, x: torch.Tensor) -> torch.Tensor:
        """Unperturbed FP32 forward pass

        Args:
            x: Layer input

        Returns:
            Layer output
        """

    @abstractmethod
    def quant_fwd(self, x: torch.Tensor) -> torch.Tensor:
        """Quantized Forward Pass
        Performs the layer operation with an additional pass through the
        currently configured quantizer

        Args:
            x: Layer input

        Returns:
            Layer output
        """

    @abstractmethod
    def approx_fwd(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate Product Forward Pass
        Performs the layer operation using the currently configured
        approximate product Lookup Table.
        Quantization is implicitly applied to the input and weights.

        Args:
            x: Layer input

        Returns:
            Layer output
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with currently selected mode applied

        Args:
            x: Layer input

        Returns:
            Layer output
        """
        # Calculate weight quantization range
        self.alpha_w = torch.max(
            torch.abs(torch.min(self.weight)),  # type ignore
            torch.abs(torch.max(self.weight)),  # type ignore
        )
        if self.mode == "approx":
            # approximate operation (always run in INT8 quant.)
            y = self.approx_fwd(x)
        elif self.quantize:
            # INT8 accurate operation
            y = self.quant_fwd(x)
        else:
            # FP32 accurate operation
            y = self.baseline_fwd(x)

        if self.noise and self.training:
            # Add noise to pre-activation output
            noise = torch.randn_like(y) * torch.std(y) * self.stdev
            y = y + noise

        return y
