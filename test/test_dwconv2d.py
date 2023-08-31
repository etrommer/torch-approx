import pytest
import torch

from torchapprox.operators.backend import dwconv2d

BATCH_SIZE = 32
configs = [
    (BATCH_SIZE, 32, 64, 64, 3, 1, 1),
    (BATCH_SIZE, 32, 64, 64, 5, 1, 2),
    (BATCH_SIZE, 32, 64, 64, 3, 2, 1),
    (BATCH_SIZE, 32, 64, 64, 5, 2, 2),
    (BATCH_SIZE, 32, 64, 64, 3, 1, 0),
    (BATCH_SIZE, 32, 64, 64, 5, 1, 0),
    (BATCH_SIZE, 32, 64, 64, 3, 2, 0),
    (BATCH_SIZE, 32, 64, 64, 5, 2, 0),
    (BATCH_SIZE, 32, 63, 65, 3, 1, 1),
    (BATCH_SIZE, 32, 63, 65, 5, 1, 2),
    (BATCH_SIZE, 32, 63, 65, 3, 2, 1),
    (BATCH_SIZE, 32, 63, 65, 5, 2, 2),
    (BATCH_SIZE, 128, 32, 32, 3, 1, 1),
    (BATCH_SIZE, 128, 32, 32, 5, 1, 2),
]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available. Skipping accelerated Conv2D kernels.",
)
@pytest.mark.parametrize("N,C,H,W,kernel_size,stride,padding", configs)
def test_dwconv2d_forward(N, C, H, W, kernel_size, stride, padding, lut):
    x = torch.randint(-128, 128, (N, C, H, W), device=torch.device("cuda")).float()
    w = torch.randint(
        -128, 128, (C, 1, kernel_size, kernel_size), device=torch.device("cuda")
    ).float()
    lut = lut.cuda()

    native = torch.nn.functional.conv2d(x, w, stride=stride, padding=padding, groups=C)
    custom = dwconv2d(x, w, lut.short(), stride=stride, padding=padding).float()
    assert torch.allclose(native, custom)
