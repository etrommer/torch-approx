#include "ta_gemm_cpu.h"

#include <stdio.h>

#include <iostream>

template <typename T> void ta_gemm_cpu(at::Tensor a, at::Tensor b, at::Tensor lut, at::Tensor res) {
    auto a_acc = a.accessor<T, 3>();
    auto b_acc = b.accessor<T, 2>();

    auto lut_acc = lut.accessor<int16_t, 2>();
    auto res_acc = res.accessor<int32_t, 3>();

#pragma omp parallel for
    for (auto batch = 0; batch < a_acc.size(0); batch++) {
        for (auto row = 0; row < a_acc.size(1); row++) {
            for (auto col = 0; col < b_acc.size(0); col++) {
                int32_t acc = 0;
                for (auto elem = 0; elem < a_acc.size(2); elem++) {
                    auto i1 = static_cast<uint8_t>(a_acc[batch][row][elem]);
                    auto i2 = static_cast<uint8_t>(b_acc[col][elem]);
                    acc += lut_acc[i1][i2];
                }
                res_acc[batch][row][col] = acc;
            }
        }
    }
}

template <typename T>
void ta_gemm_cpu_batchb(at::Tensor a, at::Tensor b, at::Tensor lut, at::Tensor res) {
    auto a_acc = a.accessor<T, 2>();
    auto b_acc = b.accessor<T, 3>();

    auto lut_acc = lut.accessor<int16_t, 2>();
    auto res_acc = res.accessor<int32_t, 3>();

#pragma omp parallel for
    for (auto batch = 0; batch < b_acc.size(0); batch++) {
        for (auto row = 0; row < a_acc.size(0); row++) {
            for (auto col = 0; col < b_acc.size(1); col++) {
                int32_t acc = 0;
                for (auto elem = 0; elem < a_acc.size(1); elem++) {
                    auto i1 = static_cast<uint8_t>(a_acc[row][elem]);
                    auto i2 = static_cast<uint8_t>(b_acc[batch][col][elem]);
                    acc += lut_acc[i2][i1];
                }
                res_acc[batch][row][col] = acc;
            }
        }
    }
}

void ta_gemm_cpu_wrapper(at::Tensor a, at::Tensor b, at::Tensor lut, at::Tensor res) {
    switch (a.scalar_type()) {
    // case torch::ScalarType::Byte:
    //     return torchapprox_cpu<uint8_t>(a,b,lut,res);
    case torch::ScalarType::Char:
        if (a.dim() == 3) {
            return ta_gemm_cpu<int8_t>(a, b, lut, res);
        } else {
            return ta_gemm_cpu_batchb<int8_t>(a, b, lut, res);
        }
    default:
        break;
    }
}

#ifndef TA_CUDA_EXTENSION
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cpu", &ta_gemm_cpu_wrapper, "torchapprox CPU backend");
}
#endif
