import pytest

from torchapprox.layers import ApproxLayer


def test_instantiate():
    with pytest.raises(TypeError):
        al = ApproxLayer()
