<div align="center">
<img src="https://github.com/etrommer/torch-approx/raw/main/docs/ta_banner.png" width="400px" height="345px" alt="Torchapprox">
<h4>
GPU-accelerated Neural Network layers using Approximate Multiplication for PyTorch
</h4>

![Build Docs](https://github.com/etrommer/torch-approx/actions/workflows/docs.yaml/badge.svg)
![Unit Tests](https://github.com/etrommer/torch-approx/actions/workflows/pytest.yaml/badge.svg)
</div>

- [1. Installation](#1-installation)
  - [1.1. Dependencies](#11-dependencies)
    - [1.1.1. Ubuntu (22.04)](#111-ubuntu-2204)
  - [1.2. Package](#12-package)
    - [1.2.1. via pip](#121-via-pip)
  - [1.3. EvoApproxLib](#13-evoapproxlib)
  - [1.4. Development](#14-development)
    - [1.4.1. Run Unit Tests](#141-run-unit-tests)
    - [1.4.2. Run Benchmarking](#142-run-benchmarking)
- [2. Usage](#2-usage)
- [3. Contributing](#3-contributing)
- [4. License](#4-license)
- [5. Credits](#5-credits)

# 1. Installation
## 1.1. Dependencies
### 1.1.1. Ubuntu (22.04)
```bash
sudo apt install libomp-dev ninja-build
```
## 1.2. Package
### 1.2.1. via pip
```bash
$ pip install git+ssh://github.com/etrommer/torch-approx
```
## 1.3. EvoApproxLib
The [EvoApprox Library of Approximate Circuits](https://github.com/ehw-fit/evoapproxlib/tree/v2022) is used as a reference and for testing in several places throughout this package. Because downloading and compiling the library is relatively time-consuming, it is not part of the automated package install. Instead, it needs to be installed by hand if required. Run:
```bash
./install_evoapprox.sh
```
to download and compile the latest version inside your current Python environment
## 1.4. Development
```bash
git clone git@github.com:etrommer/torch-approx.git
cd torch-approx
poetry install --with dev extras
poetry run pre-commit install
```
### 1.4.1. Run Unit Tests
Run Unit Tests with
```bash
poetry run pytest
```
### 1.4.2. Run Benchmarking
Microbenchmarking of kernels can be carried out with:
```bash
poetry run pytest benchmarks
```

# 2. Usage
1. Define & Train NN Model (or use a pre-trained model)
    ```python
    import torchvision.models as models

    model = models.mobilenet_v2()
    ```
2. Convert to Torchapprox
    ```python
    from torchapprox.utils.conversion import inplace_conversion

    approx_model = inplace_conversion(model)
    ```
3. Set Inference Mode
    ```python
    from torchapprox.utils.conversion import get_approx_modules
    import torchapprox.layers as tal

    approx_modules = get_approx_modules(approx_model)
    for _, m in approx_modules:
        m.inference_mode = tal.InferenceMode.APPROXIMATE
    ```
4. Set LUT (assuming EvoApproxLib has been installed [as described above](#13-evoapproxlib))
    ```python
    from torchapprox.utils.evoapprox import lut

    for _, m in approx_modules:
        m.approx_op.lut = lut('mul8s_1KV6')
    ```
    Any 256x256 numpy array can be supplied as a LUT, depending on your use case. Note that the axes have to follow this ordering:
    ```
    x = y = [0, 1, 2, ... 126, 127, -128, -127, ...-2, -1]
    ```
    where the LUT entry at position `lut[x,y]` is the approxmiate multiplication result of `x` and `y`.
5. Train
    Train your model using [a standard PyTorch Training Pipeline](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)



# 3. Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

# 4. License

`torchapprox` was created by Elias Trommer. It is licensed under the terms of the MIT license.

# 5. Credits

- `torchapprox` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).  
- Depthwise Convolution Kernels based on: [https://github.com/rosinality/depthwise-conv-pytorch](https://github.com/rosinality/depthwise-conv-pytorch)  
- This work was created as part of my Ph.D. research at Infineon Technologies Dresden  
