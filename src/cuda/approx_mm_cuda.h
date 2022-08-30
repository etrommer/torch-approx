#pragma once
#include <torch/extension.h>

void approx_mm_cuda_kernel(at::Tensor a, at::Tensor b, at::Tensor lut, at::Tensor res);
