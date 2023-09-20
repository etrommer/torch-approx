<div align="center">
<img src="https://github.com/etrommer/torch-approx/raw/main/docs/torchapprox_logo.png" width="400px" height="280px" alt="Torchapprox">
<h4>
GPU-accelerated Neural Network layers using Approximate Multiplication for PyTorch
</h4>

![Build Docs](https://github.com/etrommer/torch-approx/actions/workflows/docs.yaml/badge.svg)
![Unit Tests](https://github.com/etrommer/torch-approx/actions/workflows/pytest.yaml/badge.svg)
</div>

- [1. Summary](#1-summary)
- [2. Installation](#2-installation)
  - [2.1. Dependencies](#21-dependencies)
    - [2.1.1. Ubuntu (22.04)](#211-ubuntu-2204)
  - [2.2. Package](#22-package)
    - [2.2.1. via pip](#221-via-pip)
  - [2.3. EvoApproxLib (optional)](#23-evoapproxlib-optional)
  - [2.4. Development](#24-development)
    - [2.4.1. Run Unit Tests](#241-run-unit-tests)
    - [2.4.2. Run Benchmarking](#242-run-benchmarking)
- [3. Usage](#3-usage)
- [4. Contributing](#4-contributing)
- [5. License](#5-license)
- [6. Credits](#6-credits)

# 1. Summary
TorchApprox is a software package for the use of approximate multiplication functions in the inference/training path of a deep neural network. Approximate Multiplications don't necessarily return mathematically accurate results but can help save resources on constrained systems. Since the multiplication is a fundamental operation in Deep Neural Networks, they are not easily compatible with vanilla PyTorch. TorchApprox provides special layers that can be used to simulate arbitrary multiplication functions using either:
- a pre-computed lookup table of the desired product function. These LUTs accurately reproduce the behavior of the multiplication function, but are much slower and limited to operand bitwidths of 8x8-Bits.
- Floating-Point models of the approximate product function. These are much faster and work independently of operand bitwidth but only exist for a subset of approximate multipliers. They also don't guarantee an accurate reproduction of the product function under all circumstances. TorchApprox comes with HTP models for all signed multipliers from the [EvoApprox library](https://ehw.fit.vutbr.cz/evoapproxlib/). These can be found under: [src/torchapprox/operators/htp_models](src/torchapprox/operators/htp_models)

# 2. Installation
Installation is currently only tested on unix-like systems.
## 2.1. Dependencies
TorchApprox depends on [OpenMP](https://www.openmp.org/) for multithreaded CPU fallback kernels and on the [Ninja Build System](https://ninja-build.org/) for JIT-compilation of all kernels. [CUDA](https://en.wikipedia.org/wiki/CUDA) is required if you have want to use an nVidia GPU for model training. This should already be available if you have correctly installed [PyTorch](https://pytorch.org/get-started/locally/).

### 2.1.1. Ubuntu (22.04)
```bash
sudo apt install libomp-dev ninja-build
```
## 2.2. Package
### 2.2.1. via pip
```bash
$ pip install git+ssh://github.com/etrommer/torch-approx
```

## 2.3. EvoApproxLib (optional)
The [EvoApprox Library of Approximate Circuits](https://github.com/ehw-fit/evoapproxlib/tree/v2022) is used as a reference and for testing in several places throughout this package. Because downloading and compiling the library is relatively time-consuming, it is not part of the automated package install. Instead, it needs to be installed by hand if required. Run:
```bash
./install_evoapprox.sh
```
to download and compile the latest version inside your current Python environment

## 2.4. Development
TorchApprox uses [poetry](https://python-poetry.org/) for build and dependency management.  
To clone a version with all development dependencies, run:
```bash
git clone git@github.com:etrommer/torch-approx.git
cd torch-approx
poetry install --with dev extras
poetry run pre-commit install
```
### 2.4.1. Run Unit Tests
Run Unit Tests with
```bash
poetry run pytest
```
some unit tests will be skipped if there is no CUDA device found on your system.

### 2.4.2. Run Benchmarking
Microbenchmarking of kernels can be carried out with:
```bash
poetry run pytest benchmarks
```

# 3. Usage
This is a minimal example of how to set up an approximate model in PyTorch. The starting point can either be a custom or a pre-trained model.

1. Define NN Model  
    Any PyTorch CNN/FC model will suffice. We will use a pre-trained model here.
    ```python
    import torchvision.models as models

    model = models.mobilenet_v2()
    ```
2. Convert to Torchapprox  
    This will automatically detect all `torch.nn.Linear` and `torch.nn.Conv2d` layers in your model and convert them to `torchapprox.layers.ApproxLinear` and `torchapprox.layers.ApproxConv2d` instances. Weights, biases, configuration, etc. are retained.

    ```python
    from torchapprox.utils.conversion import inplace_conversion

    approx_model = inplace_conversion(model)
    ```
3. Set Inference Mode  

    The inference mode tells TorchApprox which product function to use. Available inference modes are:
    - `BASELINE`: Accurate, floating-point product function
    - `QUANTIZED`: Accurate, floating-point product function, but operands are passed through a quantization algorithm first (fake-quantization)
    - `NOISE`: Fake-quantization (same as `QUANTIZED`), but with Additive Gaussian Noise applied to the operation's output
    - `APPROXIMATE`: Quantize operands and apply the configured approximate product function.

    Configure your target layers for approximate inference like this:
    ```python
    from torchapprox.utils.conversion import get_approx_modules
    import torchapprox.layers as tal

    approx_modules = get_approx_modules(approx_model)
    for _, m in approx_modules:
        m.inference_mode = tal.InferenceMode.APPROXIMATE
    ```
4. Apply the desired approximate product function to each approximate module. 
   There are two ways of doing this: you can either use a Lookup table, or an HTP model. The benefits and downsides of both are briefly discussed in the [Summary](#1-summary).
   
    **Option 1:** Use LUT  
    This example assumes that EvoApproxLib has been installed [as described above](#23-evoapproxlib).
    ```python
    from torchapprox.utils.evoapprox import lut

    for _, m in approx_modules:
        m.approx_op.lut = lut('mul8s_1KV6')
    ```
    Of course, you are free to construct your own LUT if you prefer.  
    Any 256x256 numpy array can be supplied as a LUT, depending on your use case. Note that the axes have to follow this ordering:
    ```
    x = y = [0, 1, 2, ... 126, 127, -128, -127, ...-2, -1]
    ```
    where the LUT entry at position `lut[x,y]` is the approxmiate multiplication result of `x` and `y`.  

    For the sake of illustration, here is an example of how to simulate an accurate multiplier using the LUT approach:
    ```python
    import numpy as np
    
    # Construct LUT
    x = np.arange(256)
    x[x >= 128] -= 256
    xx, yy = np.meshgrid(x, x)
    my_lut = xx * yy

    # Apply LUT to approximate layers
    for _, m in approx_modules:
        m.approx_op.lut = my_lut

    import torch
    
    # Alternatively, the LUT can also be supplied as a torch tensor (required to be 16-Bit Int)
    for _, m in approx_modules:
        m.approx_op.lut = torch.from_numpy(my_lut).short()
    ```

    **Option 2:** Use HTP model
    ```python
    from torchapprox.operators.htp_models import htp_models_mul8s

    for _, m in approx_modules:
        m.fast_model = htp_models_mul8s['mul8s_1KV6']
    ```
    Note that HTP models take precedence over LUTs, i.e. if a layer is set to `APPROXIMATE` inference and both a LUT and HTP model are configured, the LUT will be ignored and the HTP model will be used. To remove HTP models, simply set them to `None`:

    ```python
    for _, m in approx_modules:
        m.fast_model = None
    ```
5. Train  
    Train your model using [a standard PyTorch Training Pipeline](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)



# 4. Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

# 5. License

`torchapprox` was created by Elias Trommer. It is licensed under the terms of the MIT license.

# 6. Credits

- `torchapprox` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).  
- Depthwise Convolution Kernels based on: [https://github.com/rosinality/depthwise-conv-pytorch](https://github.com/rosinality/depthwise-conv-pytorch)  
- This work was created as part of my Ph.D. research at Infineon Technologies Dresden  
