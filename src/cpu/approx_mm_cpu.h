#ifndef _TORCHAPPROX_CPU
#define _TORCHAPPROX_CPU

#include <torch/extension.h>

void approx_mm_cpu_wrapper(at::Tensor a, at::Tensor b, at::Tensor lut, at::Tensor res);

#endif
