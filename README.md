<div align="center">
<img src="https://github.com/etrommer/torch-approx/raw/main/docs/torchapprox_logo.png" width="400px" height="280px" alt="Torchapprox">
<h4>
GPU-accelerated Neural Network layers using Approximate Multiplication for PyTorch
</h4>

![Build Docs](https://github.com/etrommer/torch-approx/actions/workflows/docs.yaml/badge.svg)
![Unit Tests](https://github.com/etrommer/torch-approx/actions/workflows/pytest.yaml/badge.svg)
</div>

## Installation

### via pip
```bash
$ pip install torchapprox
```
### via Github
```bash
git clone https://github.com/etrommer/torch-approx
cd torch-approx
poetry install
```

## Usage

### Unit Test
Run Unit Tests with
```bash
poetry run pytest
```
### Benchmarking
```bash
poetry run pytest benchmarks
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`torchapprox` was created by Elias Trommer. It is licensed under the terms of the MIT license.

## Credits

- `torchapprox` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).  
- Depthwise Convolution Kernels based on: [https://github.com/rosinality/depthwise-conv-pytorch](https://github.com/rosinality/depthwise-conv-pytorch)  
- This work was created as part of my Ph.D. research at Infineon Technologies Dresden  
