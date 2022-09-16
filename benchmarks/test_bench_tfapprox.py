import pytest
import tensorflow as tf
from fake_approx_convolutional import FakeApproxConv2D

print(tf.__version__)

channels = [1, 2, 4, 8, 16, 32, 64]


@pytest.mark.parametrize("channels", channels)
def test_bench_tfapprox(benchmark, channels):
    approx_model = tf.keras.Sequential(
        [
            FakeApproxConv2D(
                filters=channels,
                kernel_size=(3, 3),
                mul_map_file="/home/elias/tf-approximate/tf2/examples/axmul_8x8/mul8u_2HH.bin",
                data_format="channels_first",
                padding="same",
            )
        ]
    )
    x = tf.random.uniform([128, channels, 224, 224])

    def benchmark_fn(x):
        approx_model(x)

    benchmark(benchmark_fn, x)
