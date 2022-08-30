#include "approx_mm_wrapper.h"
#include "../cpu/approx_mm_cpu.h"
#include "approx_mm_cuda.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                 \
    CHECK_CONTIGUOUS(x)

void approx_mm_cuda_wrapper(at::Tensor a, at::Tensor b, at::Tensor lut, at::Tensor res) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(lut);
    CHECK_INPUT(res);

    return approx_mm_cuda_kernel(a, b, lut, res);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &approx_mm_cuda_wrapper, "torchapprox CUDA Backend");
    m.def("matmul_cpu", &approx_mm_cpu_wrapper, "torchapprox CPU backend");
}
