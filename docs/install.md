# Installation
## Dependencies
TorchApprox relies on the [ninja build system](https://ninja-build.org/) and [OpenMP](https://www.openmp.org/) in order to JIT-compile kernels. Make sure they are available on your system.
### Ubuntu
```bash
$ sudo apt install libomp-dev ninja-build
```
## Pip Installation
```bash
$ pip install git+ssh://github.com/etrommer/torch-approx
```

## Development
If you want to help actively develop TorchApprox, clone TorchApprox from Github and make sure to install additional dependencies as well as pre-commit hooks:
```bash
git clone git@github.com:etrommer/torch-approx.git
cd torch-approx
poetry install --with dev extras
poetry run pre-commit install
```
## Unit Tests
TorchApprox uses [pytest](https://docs.pytest.org/en/7.1.x/getting-started.html) for unit testing. Unit tests can be run with:
```bash
poetry run pytest test
```
If a CUDA device is detected, tests will be run on CPU and GPU. Otherwise, tests run on CPU only.

## Micro-Benchmarking
TorchApprox uses the [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/) plugin to perform benchmarking of low-level kernels. Benchmarking can be run with:
```bash
poetry run pytest benchmarks
```
