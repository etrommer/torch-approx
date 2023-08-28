# pylint: disable=missing-module-docstring
import enum
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, no_type_check

import torch
import torch.ao.quantization as tq

from torchapprox.operators import LUTGeMM

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class InferenceMode(enum.Enum):
    """
    Layer inference mode. Can be any of:
    - `quant`: Run inference using the layer's quantizer
    - `approx`: Run inference using approximate product LUT
    - `noise`: Run inference that is perturbed with additive Gaussian noise
    """

    QUANTIZED = "Quantized Mode"
    NOISE = "Noise Mode"
    APPROXIMATE = "Approximate Mode"


class ApproxLayer(ABC):
    """
    Derivable Abstract Base Class for implementing Approximate Neural Network layers
    """

    def __init__(self, qconfig: Optional[tq.QConfig] = None):
        self.approx_op: LUTGeMM = LUTGeMM()
        self.inference_mode: InferenceMode = InferenceMode.QUANTIZED
        self.fast_model: Optional[Callable] = None

        self._stdev: torch.Tensor = torch.tensor([0.0])
        self._mean: torch.Tensor = torch.tensor([0.0])

    @staticmethod
    def default_qconfig() -> tq.QConfig:
        act_qconfig = tq.FakeQuantize.with_args(
            observer=tq.HistogramObserver,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            quant_min=0,
            quant_max=127,
        )
        weight_qconfig = tq.FakeQuantize.with_args(
            observer=tq.HistogramObserver,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            quant_min=-128,
            quant_max=127,
        )
        return tq.QConfig(activation=act_qconfig, weight=weight_qconfig)

    @property
    def stdev(self) -> float:
        """
        Perturbation Error Relative Standard Deviation

        Returns:
            Currently configured perturbation standard deviation
        """
        return self._stdev.item()

    @stdev.setter
    def stdev(self, val: float):
        self._stdev = torch.tensor([val], device=self.weight.device)  # type: ignore

    @property
    def mean(self) -> float:
        """
        Perturbation Error mean

        Returns:
            Currently configured perturbation mean
        """
        return self._mean.item()

    @mean.setter
    def mean(self, val: float):
        self._mean = torch.tensor([val], device=self.weight.device)  # type: ignore

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

    @abstractmethod
    def quant_fwd(
        self, x_q: torch.FloatTensor, w_q: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Quantized Forward Pass
        Performs the layer operation with an additional pass through the
        currently configured quantizer.

        `x_q and w_q are expected to be **fake-quantized** tensors, i.e. floats that are
        discretized to a set of values, but not converted to actual their integer
        representation.

        Args:
            x_q: Fake-quantized activations
            w_q: Fake-quantized weights

        Returns:
            Layer output
        """

    @abstractmethod
    def approx_fwd(
        self,
        x_q: torch.CharTensor,
        w_q: torch.CharTensor,
        x_scale: torch.Tensor,
        x_zero_point: torch.Tensor,
        w_scale: torch.Tensor,
        w_zero_point: torch.Tensor,
    ):
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
    def noise_fwd(
        self, x_q: torch.FloatTensor, w_q: torch.FloatTensor
    ) -> torch.FloatTensor:
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
        y = self.quant_fwd(x_q, w_q)
        if self.training:
            noise = torch.randn_like(y) * torch.std(y) * self.stdev + self.mean
            y = y + noise
        return y

    @no_type_check
    def forward(
        self,
        x_q: torch.Tensor,
        x_scale: Optional[torch.Tensor] = None,
        x_zero_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with currently selected mode applied

        Args:
            x: Layer input

        Returns:
            Layer output
        """
        assert hasattr(
            self, "weight_fake_quant"
        ), "QAT nodes not replaced. Run `prepare_qat` first."

        w_q = self.weight_fake_quant(self.weight)
        if self.inference_mode == InferenceMode.NOISE:
            y_q = self.noise_fwd(x_q)
        elif self.inference_mode == InferenceMode.APPROXIMATE:
            assert (x_scale is not None) and (
                x_zero_point is not None
            ), "Received no activation quantization information during approximate forward pass"
            w_scale = self.weight_fake_quant.scale
            w_zero_point = self.weight_fake_quant.zero_point
            w_scale = self.weight_fake_quant.scale
            y_q = self.approx_fwd(
                x_q, w_q, x_scale, x_zero_point, w_scale, w_zero_point
            )
        else:
            y_q = self.quant_fwd(x_q, w_q)

        if self.bias is not None:
            y_q = y_q + self.bias

        return y_q
