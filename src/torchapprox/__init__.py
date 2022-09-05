"""
TorchApprox Base package
"""
# isort: skip_file
# read version from installed package
from importlib.metadata import version

from .torchapprox import ApproxMM, approx
from .layers import *
from .quantizers import *

__version__ = version("torchapprox")
