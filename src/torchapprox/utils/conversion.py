# pylint: disable=missing-module-docstring
from typing import List, Optional, Tuple

import torch

import torchapprox.layers as tal


def wrap_quantizable(
    net: torch.nn.Module,
    wrappable_layers: Optional[List[tal.ApproxLayer]] = None,
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
    if not wrappable_layers:
        wrappable_layers = [torch.nn.Linear, torch.nn.Conv2d]

    replace_list = []

    def find_replacable_modules(parent_module):
        for name, child_module in parent_module.named_children():
            if any([isinstance(child_module, t) for t in wrappable_layers]):
                replace_list.append((parent_module, name))
        for child in parent_module.children():
            find_replacable_modules(child)

    find_replacable_modules(net)

    for parent, name in replace_list:
        orig_layer = getattr(parent, name)
        wrapped = tal.ApproxWrapper(orig_layer)
        setattr(parent, name, wrapped)
    return net


def get_approx_modules(net: torch.nn.Module) -> List[Tuple[str, tal.ApproxLayer]]:
    """
    Retrieve all approximate layers from a model

    Args:
        net: PyTorch neural network model

    Returns:
        A list of tuples with name and reference to each Approximate layer instance in the model
    """
    return [(n, m) for n, m in net.named_modules() if isinstance(m, tal.ApproxLayer)]
