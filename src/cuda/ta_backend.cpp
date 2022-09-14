#include "ta_backend.h"
#include "../cpu/ta_gemm_cpu.h"
#include "ta_gemm_cuda.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                 \
    CHECK_CONTIGUOUS(x)

void ta_gemm_cuda_wrapper(at::Tensor a, at::Tensor b, at::Tensor lut, at::Tensor res) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(lut);
    CHECK_INPUT(res);

    return ta_gemm_cuda_launch(a, b, lut, res);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &ta_gemm_cuda_wrapper, "ApproxGeMM (CUDA)");
    m.def("matmul_cpu", &ta_gemm_cpu_wrapper, "ApproxGeMM (CPU)");
}
