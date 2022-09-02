# pylint: disable=missing-module-docstring
from abc import ABC, abstractmethod

import torch


class ApproxQuantizer(torch.nn.Module, ABC):
    """
    Abstract Quantizer interface definition
    """

    def __init__(self, bitwidth: int = 8):
        torch.nn.Module.__init__(self)
        self._bitwidth = bitwidth

    @abstractmethod
    def fake_quant(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply Fake Quantization to a tensor of floats

        Args:
            x: Floating-point input

        Returns:
            Fake-quantized output
        """

    @abstractmethod
    def quantize(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply quantization operation to input tensor

        Args:
            x: Floating-point input

        Returns:
            Output quantized to Integer range
        """

    @property
    def bitwidth(self) -> int:
        """
        The configured bitwidth

        Returns:
            Currently configured bitwidth
        """
        return self._bitwidth

    @property
    @abstractmethod
    def scale_factor(self) -> float:
        """
        Scale factor of quantizer

        Returns:
            The value with which a float tensor is multiplied
            in order to scale it to Integer numerical range
        """

    @property
    def int_max(self) -> int:
        """
        Highest value of an Integer according for current bitwidth

        Returns:
            Max Integer value for current bitwidth
        """
        return 2 ** (self.bitwidth - 1) - 1
