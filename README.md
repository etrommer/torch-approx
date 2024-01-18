<div align="center">
<img src="https://github.com/etrommer/torch-approx/raw/main/docs/images/ta_banner.png" width="400px" height="345px" alt="Torchapprox">
<h4>
GPU-accelerated Neural Network layers using Approximate Multiplication for PyTorch
</h4>

![Build Docs](https://github.com/etrommer/torch-approx/actions/workflows/docs.yaml/badge.svg)
![Unit Tests](https://github.com/etrommer/torch-approx/actions/workflows/pytest.yaml/badge.svg)
</div>

# 1. Documentation
For detailed installation and usage guidelines, please refer to this project's [documentation](https://etrommer.de/torch-approx)

# 2. Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

# 3. License

`torchapprox` was created by Elias Trommer. It is licensed under the terms of the MIT license.

# 4. Credits

- `torchapprox` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).  
- Depthwise Convolution Kernels based on: [https://github.com/rosinality/depthwise-conv-pytorch](https://github.com/rosinality/depthwise-conv-pytorch)  
- This work was created as part of my Ph.D. research at Infineon Technologies Dresden  

# 5. Citation
If you use TorchApprox in your work, please cite it as:
```latex
@inproceedings{trommer23torchapprox,
  author       = {Elias Trommer and
                  Bernd Waschneck and
                  Akash Kumar},
  editor       = {Maksim Jenihhin and
                  Hana Kub{\'{a}}tov{\'{a}} and
                  Nele Metens and
                  Jaan Raik and
                  Foisal Ahmed and
                  Jan Belohoubek},
  title        = {High-Throughput Approximate Multiplication Models in PyTorch},
  booktitle    = {26th International Symposium on Design and Diagnostics of Electronic
                  Circuits and Systems, {DDECS} 2023, Tallinn, Estonia, May 3-5, 2023},
  pages        = {79--82},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/DDECS57882.2023.10139366},
  doi          = {10.1109/DDECS57882.2023.10139366},
}
```
