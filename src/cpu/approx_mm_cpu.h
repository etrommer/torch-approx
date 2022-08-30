#pragma once

#include <torch/extension.h>

void approx_mm_cpu_wrapper(at::Tensor a, at::Tensor b, at::Tensor lut, at::Tensor res);
