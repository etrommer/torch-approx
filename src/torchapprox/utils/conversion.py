# pylint: disable=missing-module-docstring
from typing import Dict, Optional, Type

import torch

import torchapprox.layers as tal


def inplace_conversion(
    net: torch.nn.Module,
    layer_mappings: Optional[Dict[Type[torch.nn.Module], Type[tal.ApproxLayer]]] = None,
) -> torch.nn.Module:
    """
    Performs in-place upgrade of layers in a vanilla PyTorch network to TorchApprox
    approximate layer implementations

    Args:
        net: PyTorch neural network model
        layer_mappings: Mapping Dict where the keys correspond to regular PyTorch layers and
            values correspond to TorchApprox layers they are replaced with

    Returns:
        An identical model with target layers replaced by Approximate Layer implementations
    """
    if layer_mappings is None:
        layer_mappings = {
            torch.nn.Conv2d: tal.ApproxConv2d,
            torch.nn.Linear: tal.ApproxLinear,
        }

    def replace_module(parent_module, base_type, approx_type):
        for name, child_module in parent_module.named_children():
            for child in parent_module.children():
                replace_module(child, base_type, approx_type)
            if isinstance(child_module, base_type):
                setattr(parent_module, name, approx_type.from_super(child_module))

    for base_type, approx_type in layer_mappings.items():
        replace_module(net, base_type, approx_type)

    return net
