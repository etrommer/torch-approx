import pytest
import tensorflow as tf
from conftest import BATCH_SIZE, CONV2_DIM, channels
from fake_approx_convolutional import FakeApproxConv2D


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
    tf.config.experimental.set_synchronous_execution(True)
    x = tf.random.normal([BATCH_SIZE, channels, CONV2_DIM, CONV2_DIM])

    def benchmark_fn(x):
        y = approx_model(x)
        return y

    benchmark(benchmark_fn, x)
