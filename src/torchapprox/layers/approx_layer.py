# pylint: disable=missing-module-docstring
import enum
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, no_type_check, Union
from dataclasses import dataclass

import torch
import torch.ao.quantization as tq
import numpy as np
import numpy.typing as npt

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


@dataclass
class QuantizationParameters:
    x_scale: torch.FloatTensor
    x_zero_point: torch.FloatTensor
    w_scale: torch.FloatTensor
    w_zero_point: torch.FloatTensor


@dataclass
class TracedGeMMInputs:
    features: Optional[torch.FloatTensor]
    weights: Optional[torch.FloatTensor]

    def trace(self, x_q: torch.Tensor, w_q: torch.Tensor):
        if self.features is None:
            self.features = x_q.detach().cpu().float()
        else:
            self.features = torch.cat([self.features, x_q])

        if self.weights is None:
            self.weights = w_q.detach().cpu().float()


class ApproxLayer(ABC):
    """
    Derivable Abstract Base Class for implementing Approximate Neural Network layers
    """

    def __init__(
        self, qconfig: Optional[tq.QConfig] = None, learnable_noise: bool = False
    ):
        self.inference_mode: InferenceMode = InferenceMode.QUANTIZED

        self._lut: Optional[torch.ShortTensor] = None
        self.lut = self.accurate_lut()

        self.htp_model: Optional[Callable] = None
        self.traced_inputs: Optional[TracedGeMMInputs] = None

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
        self, x: torch.FloatTensor, w: torch.FloatTensor
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
        x: torch.CharTensor,
        w: torch.CharTensor,
        quant_params: QuantizationParameters,
    ):
        """Approximate Product Forward Pass
        Performs the layer operation using the currently configured
        approximate product Lookup Table.

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
        x: torch.Tensor,
        x_scale: Optional[torch.Tensor] = None,
        x_zero_point: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
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

        w = self.weight_fake_quant(self.weight)
        if self.inference_mode == InferenceMode.NOISE:
            y = self.noise_fwd(x, w)
        elif self.inference_mode == InferenceMode.APPROXIMATE:
            assert (x_scale is not None) and (
                x_zero_point is not None
            ), "Received no activation quantization information during approximate forward pass"
            assert (
                len(x_scale) == 1 and len(x_zero_point) == 1
            ), "Per-channel quantization only supported for weights"
            quant_params = QuantizationParameters(
                x_scale,
                x_zero_point,
                self.weight_fake_quant.scale,
                self.weight_fake_quant.zero_point,
            )
            y = self.approx_fwd(x, w, quant_params)
        else:
            y = self.quant_fwd(x, w)

        if bias is not None:
            y = y + bias
        elif self.bias is not None:
            y = y + self.bias

        return y
