# pylint: disable=missing-module-docstring
from typing import List, Optional, Tuple

import torch

import torchapprox.layers as tal
import torch.ao.quantization as tq


def wrap_quantizable(
    net: torch.nn.Module,
    wrappable_layers: Optional[List[tal.ApproxLayer]] = None,
    qconfig: Optional[tq.QConfig] = None,
) -> torch.nn.Module:
    """
    Performs in-place upgrade of layers in a vanilla PyTorch network to TorchApprox
    approximate layer implementations. Regular insertion of quant/dequant stubs does not work
    because the activation quantization parameters are required _inside_ the quantized layer.

    Args:
        net: PyTorch neural network model
        wrappable_layers: Layer types to be wrapped

    Returns:
        An identical model with target layers replaced by Approximate Layer implementations
    """
    if not wrappable_layers:
        wrappable_layers = [torch.nn.Linear, torch.nn.Conv2d]

    replace_list = []

    def find_replacable_modules(parent_module):
        if isinstance(parent_module, tal.ApproxWrapper):
            return
        for name, child_module in parent_module.named_children():
            if any([isinstance(child_module, t) for t in wrappable_layers]):
                replace_list.append((parent_module, name))
        for child in parent_module.children():
            find_replacable_modules(child)

    find_replacable_modules(net)

    for parent, name in replace_list:
        orig_layer = getattr(parent, name)
        wrapped = tal.ApproxWrapper(orig_layer, qconfig)
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
