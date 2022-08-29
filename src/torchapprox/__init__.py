"""
TorchApprox Base package
"""
# read version from installed package
from importlib.metadata import version

from .torchapprox import ApproxMM, approx

__version__ = version("torchapprox")
