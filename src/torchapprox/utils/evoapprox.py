"""
Utilities for working with the EvoApprox Libray
"""
import importlib
import pkgutil
import re
from typing import Any, List, Tuple

import evoapproxlib as eal
import numpy as np


def module_names(filter_str: str = "") -> List[str]:
    """
    List all available modules in EvoApproxLib

    Args:
        filter_str: Only return moduls whose name contains a certain substring, i.e. `mul8s`. Defaults to "".

    Raises:
        ImportError: Loading of library failed

    Returns:
        List of (filtered) available module names
    """

    modules = [m.name for m in pkgutil.iter_modules(eal.__path__)]
    return [m for m in modules if filter_str in m]


def attribute(multiplier_name: str, attr_name: str) -> Any:
    """
    Read Attribute from EvoApprox module

    Args:
        multiplier_name: Name of the target multiplier
        attr_name: Name of the target attribute

    Returns:
        Target attribute value
    """
    mul = load_multiplier(multiplier_name)
    return getattr(mul, attr_name)


def load_multiplier(multiplier_name):
    """
    Try loading a multiplier module from the EvoApprox library

    Args:
        multiplier_name: String with name of target multiplier

    Raises:
        ImportError: Multiplier module could not be loaded

    Returns:
        EvoApprox module
    """
    try:
        multiplier = importlib.import_module(f"evoapproxlib.{multiplier_name}")
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError(
            f"Importing {multiplier_name} from EvoApproxLib failed. Possibly due to unavailable EvoApproxLib."
        ) from exc
    return multiplier


def approx_multiplication(multiplier_name, range_x, range_y, signed, bitwidth):
    """
    Calculate approximate multiplication result of supplied
    approximate multiplier along a given range.

    Args:
        multiplier: The `evoapprox` class to evaluate
        range_x: Iterable of first operand range
        range_y: Iterable of second operand range
        signed: Signed or unsigned multiplication
        bitwidth: Bitwidth of approximate multiplier

    Returns:
        Tuple of:
        - 2D Array of the approximate multiplication result across the
          Cartesian product of the supplied input operand ranges
        - 2D Array of x operands
        - 2D Array of y operands
    """

    multiplier = load_multiplier(multiplier_name)
    x_2d, y_2d = np.meshgrid(range_x, range_y, indexing="ij")
    approx = np.vectorize(multiplier.calc)
    res = approx(x_2d, y_2d)
    if signed:
        res[res >= 2 ** (2 * bitwidth - 1)] -= 2 ** (2 * bitwidth)
    return res, x_2d, y_2d


def error_map(multiplier_name: str) -> np.ndarray:
    """
    Generate the error map for a given multiplier,
    i.e. the difference between accurate and approximate result across the entire input space

    Args:
        multiplier_name: Name for target multiplier

    Returns:
        Error Map for multiplier input space
    """
    # Infer bitwidth from multiplier name
    bitwidth = bitwidth_from_name(multiplier_name)
    # Infer signedness from multiplier name
    signed = signedness_from_name(multiplier_name)

    if signed:
        x = np.arange(-(2 ** (bitwidth - 1)), 2 ** (bitwidth - 1))
    else:
        x = np.arange(2**bitwidth)

    res, x_2d, y_2d = approx_multiplication(multiplier_name, x, x, signed, bitwidth)
    return res - (x_2d * y_2d)


def error_distribution(
    multiplier_name: str,
    fan_in: int = 1,
) -> Tuple[Any, Any]:
    """
    Calculate the Mean and Standard Deviation of an EvoApprox multiplier

    Args:
        multiplier_name: Name of the desired multiplier, e.g. `mul8s_1KV6`
        fan_in: Compensates the reduced standard deviation from repeated sampling
            when a large number of multiplications is accumulated in a neuron.
            Defaults to `1`.
    Returns:
        Tuple of:
        - Mean of the approximate multiplication error, scaled to the neuron output
        - Standard deviation of approximate multiplication error, scaled to the neuron output
    """
    error = error_map(multiplier_name)
    return np.mean(error) * fan_in, np.std(error) * np.sqrt(fan_in)


def bitwidth_from_name(multiplier_name: str) -> int:
    """
    Helper function to extract the bitwidth from a given multiplier name

    Args:
        multiplier_name: Target multiplier name

    Returns:
        The multipliers bitwidth, i.e. 8 for `mul8s_...`
    """
    multiplier_prefix = multiplier_name.split("_")[0]
    return int(re.sub("[^0-9]", "", multiplier_prefix))


def signedness_from_name(multiplier_name: str) -> bool:
    """
    Helper function to extract the signedness for a given multiplier name

    Args:
        multiplier_name: Target multiplier name

    Returns:
        The multipliers signedness, i.e. True for `mul8s_...`
    """
    multiplier_prefix = multiplier_name.split("_")[0]
    return multiplier_prefix[-1] == "s"


def lut(
    multiplier_name: str,
) -> np.ndarray:
    """
    Generate Lookup-Table for supplied EvoApprox multiplier

    Args:
        multiplier_name: Name of the desired multiplier, e.g. `mul8s_1KV6`
        bitwidth: Bitwidth of the multiplier.
            If not supplied, this is inferred from the `multiplier_name`. Defaults to None.
        signed: Signedness of the multiplier.
            If not supplied, this is inferred from the `multiplier_name`. Defaults to None.

    Raises:
        ImportError: Raised if the supplied multiplier name could not be loaded from the evoapprox library.

    Returns:
        Lookup table of the Approximate Multiplication result across the input operand range.
    """

    # Infer bitwidth from multiplier name
    bitwidth = bitwidth_from_name(multiplier_name)
    # Infer signedness from multiplier name
    signed = signedness_from_name(multiplier_name)

    x = np.arange(2**bitwidth)

    # Build signed LUT so that values follow ordering of unsigned Integers
    # speeds up inference because indices don't have to be converted
    x[x >= (2 ** (bitwidth - 1))] -= 2**bitwidth
    # TESTING: Does this need to be run for unsigned multipliers?

    if not signed:
        x = np.abs(x)

    amul_lut, _, _ = approx_multiplication(multiplier_name, x, x, signed, bitwidth)

    if not signed:
        amul_lut[128:] *= -1
        amul_lut[:, 128:] *= -1

    return amul_lut
