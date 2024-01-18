import numpy as np
import pytest
import torch
import torchvision.models as models

networks = [
    "mobilenet_v2",
    "effcientnet_b0",
    "vgg16",
    "alexnet",
    "resnet18",
    "resnet50",
]


@pytest.fixture
def lut():
    """Create accurate 8x8-Bit Lookup Table

    Returns:
        This populates a lookup table with accurate multiplication results
        for the operand range [-128, 127]
        LUT is created so that binary representation of
        the operand is monotically increasing when cast to an unsigned number, i.e.:
        `x = y = [0, 1,...,127, -128, -127,...-1]`
        This is done to simplify the conversion of the operand to an (unsigned)
        array index in the kernel later on.
    """
    x = np.arange(256)
    x[x >= 128] -= 256
    xx, yy = np.meshgrid(x, x)
    return torch.from_numpy(xx * yy).int()


@pytest.fixture(params=networks)
def bench_architecture(request):
    if request.param == "effcientnet_b0":
        model = models.efficientnet_b0()
    if request.param == "mobilenet_v2":
        model = models.mobilenet_v2()
    elif request.param == "vgg16":
        model = models.vgg16()
    elif request.param == "alexnet":
        model = models.alexnet()
    elif request.param == "resnet18":
        model = models.resnet18()
    elif request.param == "resnet50":
        model = models.resnet50()
    # model.eval()
    return model
