#pragma once
#include <torch/extension.h>

void ta_gemm_cuda_wrapper(at::Tensor a, at::Tensor b, at::Tensor lut, at::Tensor res);
