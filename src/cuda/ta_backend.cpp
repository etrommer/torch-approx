#include "ta_backend.h"
#include "../cpu/ta_gemm_cpu.h"
#include "ta_dwconv.h"
#include "ta_gemm_cuda.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                 \
    CHECK_CONTIGUOUS(x)

torch::Tensor ta_dwconv2d_wrapper(const torch::Tensor &input, const torch::Tensor &kernel, int up_h,
                                  int up_w, int down_h, int down_w, int pad_h0, int pad_h1,
                                  int pad_w0, int pad_w1, bool forward) {
    CHECK_CUDA(input);
    CHECK_CUDA(kernel);

    return ta_dwconv2d_launch(input, kernel, up_h, up_w, down_h, down_w, pad_h0, pad_h1, pad_w0,
                              pad_w1, forward);
}

torch::Tensor ta_dwconv2d_small_wrapper(const torch::Tensor &input, const torch::Tensor &kernel,
                                        int up_h, int up_w, int down_h, int down_w, int pad_h,
                                        int pad_w, bool forward) {
    CHECK_CUDA(input);
    CHECK_CUDA(kernel);

    return ta_dwconv2d_small_launch(input, kernel, up_h, up_w, down_h, down_w, pad_h, pad_w,
                                    forward);
}

void ta_gemm_cuda_wrapper(at::Tensor a, at::Tensor b, at::Tensor lut, at::Tensor res) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(lut);
    CHECK_INPUT(res);

    return ta_gemm_cuda_launch(a, b, lut, res);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("use_dwconv2d_small", &use_dwconv2d_small, "check availability of dwconv2d small");
    m.def("dwconv2d", &ta_dwconv2d_wrapper, "dwconv2d (CUDA)");
    m.def("dwconv2d_small", &ta_dwconv2d_small_wrapper, "dwconv2d small (CUDA)");
    m.def("matmul_cuda", &ta_gemm_cuda_wrapper, "ApproxGeMM (CUDA)");
    m.def("matmul_cpu", &ta_gemm_cpu_wrapper, "ApproxGeMM (CPU)");
}
