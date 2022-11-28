# pylint: disable=missing-module-docstring
import enum
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, no_type_check

import torch

from torchapprox.operators import LUT
from torchapprox.quantizers import MinMaxQuant, PACTQuant

if TYPE_CHECKING:
    from torchapprox.quantizers import ApproxQuantizer

logger = logging.getLogger(__name__)


class InferenceMode(enum.Enum):
    """
    Layer inference mode. Can be any of:
    - `base`: Run inference as unperturbed FP32 baseline
    - `quant`: Run inference using the layer's quantizer
    - `approx`: Run inference using approximate product LUT
    - `noise`: Run inference that is perturbed with additive Gaussian noise
    """

    BASELINE = "Baseline Mode"
    QUANTIZED = "Quantized Mode"
    NOISE = "Noise Mode"
    APPROXIMATE = "Approximate Mode"


class ApproxLayer(ABC):
    """
    Derivable Abstract Base Class for implementing Approximate Neural Network layers
    """

    def __init__(self):
        self.x_quantizer: "ApproxQuantizer" = PACTQuant()
        self.w_quantizer: "ApproxQuantizer" = MinMaxQuant()
        self.approx_op: LUT = LUT()
        self.inference_mode: InferenceMode = InferenceMode.BASELINE
        self.fast_model: Optional[Callable] = None

        self._stdev: torch.nn.Paramter = torch.nn.Parameter(
            torch.tensor(0.0), requires_grad=True
        )

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

    @property
    @abstractmethod
    def fan_in(self) -> int:
        """
        Number of incoming connections for a neuron in this layer
        """

    @property
    @abstractmethod
    def opcount(self) -> int:
        """
        Number of multiplications for a single
        forward pass of this layer
        """

    @staticmethod
    @abstractmethod
    def from_super(cls_instance):
        """
        Create upgraded superclass instance.
        This constructs an approximate layer instance using the configuration
        of a vanilla torch layer implementation.
        """

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

    @no_type_check
    def noise_fwd(self, x: torch.Tensor) -> torch.Tensor:
        """Quantized Forward Pass that is perturbed
        with Gaussian Noise

        The standard deviation of the additive noise
        is derived from the `stdev`parameter and scaled
        with the standard deviation of the current batch

        Args:
            x: Layer input

        Returns:
            Layer output
        """
        y = self.quant_fwd(x)
        if self.training:
            noise = torch.randn_like(y) * torch.std(y) * self.stdev
            y = y + noise
        return y

    @no_type_check
    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with currently selected mode applied

        Args:
            x: Layer input

        Returns:
            Layer output
        """
        if self.inference_mode == InferenceMode.BASELINE:
            # FP32 accurate operation
            y = self.baseline_fwd(x)
        elif self.inference_mode == InferenceMode.QUANTIZED:
            # INT8 accurate operation
            y = self.quant_fwd(x)
        elif self.inference_mode == InferenceMode.APPROXIMATE:
            y = self.approx_fwd(x)
        elif self.inference_mode == InferenceMode.NOISE:
            y = self.noise_fwd(x)

        if bias is not None:
            y = y + bias

        return y
