#pragma once
#include <torch/extension.h>

void ta_gemm_cuda_launch(at::Tensor a, at::Tensor b, at::Tensor lut, at::Tensor res);
