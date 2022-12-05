"""
Utilities for working with the EvoApprox Libray
"""
import importlib
import re
from typing import Optional

import numpy as np


def approx_multiplication(multiplier, range_x, range_y, signed, bitwidth):
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
        Array of the approximate multiplication result across the
        Cartesian product of the supplied input operand ranges
    """
    x_2d, y_2d = np.meshgrid(range_x, range_y, indexing="ij")
    approx = np.vectorize(multiplier.calc)
    res = approx(x_2d, y_2d)
    if signed:
        res[res >= 2 ** (2 * bitwidth - 1)] -= 2 ** (2 * bitwidth)
    return res


def lut(
    multiplier_name: str, bitwidth: Optional[int] = None, signed: Optional[bool] = None
):
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
    try:
        multiplier = importlib.import_module(f"evoapproxlib.{multiplier_name}")
    except (ModuleNotFoundError, ImportError) as exc:
        raise ImportError(
            f"Importing {multiplier_name} from EvoApproxLib failed. Possibly due to unavailable EvoApproxLib."
        ) from exc

    multiplier_prefix = multiplier_name.split("_")[0]
    if bitwidth is None:
        # Infer bitwidth from multiplier name
        bitwidth = int(re.sub("[^0-9]", "", multiplier_prefix))
    if signed is None:
        # Infer signedness from multiplier name
        signed = multiplier_prefix[-1] == "s"

    x = np.arange(2**bitwidth)

    # Build signed LUT so that values follow ordering of unsigned Integers
    # speeds up inference because indices don't have to be converted
    x[x >= (2 ** (bitwidth - 1))] -= 2**bitwidth

    if not signed:
        x = np.abs(x)

    amul_lut = approx_multiplication(multiplier, x, x, signed, bitwidth)

    if not signed:
        amul_lut[128:] *= -1
        amul_lut[:, 128:] *= -1

    return amul_lut
