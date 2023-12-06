#include "ta_gemm_cuda.h"

const auto BLOCK_SIZE = 16;
const auto K = 5;

__device__ inline int32_t DRUM(int16_t op1, int16_t op2) {
    if (op1 == 0 || op2 == 0)
        return 0;
    if (op1 == -1)
        return -op2;
    if (op2 == -1)
        return -op1;

    // Sign extraction
    const bool sgn1 = op1 < 0;
    const bool sgn2 = op2 < 0;

    uint32_t abs1 = sgn1 ? -op1 : op1;
    uint32_t abs2 = sgn2 ? -op2 : op2;

    // Find leading one
    const auto lead1_1 = 31 - __clz(abs1);
    const auto lead1_2 = 31 - __clz(abs2);

    // Mask with the lowest `k` Bits set, zero otherwise
    const auto mask = (1 << K) - 1;
    if (lead1_1 > K) {
        // Truncate to the most-significant `k` bits
        abs1 &= (mask << (lead1_1 - K + 1));
        // Always set lowest non-truncated Bit position to 1
        abs1 |= (1 << (lead1_1 - K + 1));
    }
    if (lead1_2 > K) {
        abs2 &= (mask << (lead1_2 - K + 1));
        abs2 |= (1 << (lead1_2 - K + 1));
    }

    // This derives from the hardware implementation in that
    // we perform a regular multiplication instead of
    // adding and shifting to keep things simple.
    // The result is the same, however, because the approximation
    // has already been applied to the operands at this point.
    auto y0 = abs1 * abs2;
    auto y = (sgn1 ^ sgn2) ? -y0 : y0;

    return y;
}

__device__ inline int32_t mitchell_trunc(int16_t op1, int16_t op2) {
    // Same as DRUM, only that the lowest non-truncated Bit position is not
    // de-biased by setting it to one.
    if (op1 == 0 || op2 == 0)
        return 0;
    if (op1 == -1)
        return -op2;
    if (op2 == -1)
        return -op1;

    // Sign extraction
    const bool sgn1 = op1 < 0;
    const bool sgn2 = op2 < 0;

    uint32_t abs1 = sgn1 ? -op1 : op1;
    uint32_t abs2 = sgn2 ? -op2 : op2;

    // Find leading one
    const auto lead1_1 = 31 - __clz(abs1);
    const auto lead1_2 = 31 - __clz(abs2);

    // Mask with the lowest `k` Bits set, zero otherwise
    const auto mask = (1 << K) - 1;
    if (lead1_1 > K) {
        // Truncate to the most-significant `k` bits
        abs1 &= (mask << (lead1_1 - K + 1));
    }
    if (lead1_2 > K) {
        abs2 &= (mask << (lead1_2 - K + 1));
    }

    auto y0 = abs1 * abs2;
    auto y = (sgn1 ^ sgn2) ? -y0 : y0;

    return y;
}

template <typename scalar_t>
__device__ inline int32_t lut_operator(cudaTextureObject_t tex, scalar_t idx1, scalar_t idx2) {
    auto i1 = static_cast<uint8_t>(idx1);
    auto i2 = static_cast<uint8_t>(idx2);
    auto idx = (i1 << 8) | i2;
    return tex1Dfetch<int32_t>(tex, idx);
}

template <typename scalar_t>
__global__ void
ta_gemm_kernel(cudaTextureObject_t tex,
               const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> a,
               const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b,
               torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> res) {
    __shared__ scalar_t a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t b_shared[BLOCK_SIZE][BLOCK_SIZE];

    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto batch = blockIdx.z * blockDim.z + threadIdx.z;

    int32_t acc = 0;

    for (auto tile_offset = 0; tile_offset < (a.size(2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
         tile_offset++) {
        if (tile_offset * BLOCK_SIZE + threadIdx.x < a.size(2) && row < a.size(1)) {
            a_shared[threadIdx.y][threadIdx.x] =
                a[batch][row][tile_offset * BLOCK_SIZE + threadIdx.x];
        } else {
            a_shared[threadIdx.y][threadIdx.x] = 0;
        }

        if (tile_offset * BLOCK_SIZE + threadIdx.y < b.size(1) && col < b.size(0)) {
            b_shared[threadIdx.x][threadIdx.y] = b[col][tile_offset * BLOCK_SIZE + threadIdx.y];
        } else {
            b_shared[threadIdx.x][threadIdx.y] = 0;
        }

        __syncthreads();

#pragma unroll
        for (auto n = 0; n < BLOCK_SIZE; n++) {
            auto i1 = a_shared[threadIdx.y][n];
            auto i2 = b_shared[threadIdx.x][n];
            /* auto val = lut_operator<uint8_t>(tex, i1, i2);*/
            auto val = mitchell_trunc((int16_t)i1, (int16_t)i2);
            acc += val;
        }
        __syncthreads();
    }

    if (batch < res.size(0) && row < res.size(1) && col < res.size(2)) {
        res[batch][row][col] = acc;
    }
}

template <typename scalar_t>
__global__ void
ta_gemm_kernel_batchb(cudaTextureObject_t tex,
                      const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> a,
                      const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> b,
                      torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> res) {
    __shared__ scalar_t a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t b_shared[BLOCK_SIZE][BLOCK_SIZE];

    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    const auto batch = blockIdx.z * blockDim.z + threadIdx.z;

    int32_t acc = 0;

    for (auto tile_offset = 0; tile_offset < (a.size(1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
         tile_offset++) {
        if (tile_offset * BLOCK_SIZE + threadIdx.x < a.size(1) && row < a.size(0)) {
            a_shared[threadIdx.y][threadIdx.x] = a[row][tile_offset * BLOCK_SIZE + threadIdx.x];
        } else {
            a_shared[threadIdx.y][threadIdx.x] = 0;
        }

        if (tile_offset * BLOCK_SIZE + threadIdx.y < b.size(2) && col < b.size(1)) {
            b_shared[threadIdx.x][threadIdx.y] =
                b[batch][col][tile_offset * BLOCK_SIZE + threadIdx.y];
        } else {
            b_shared[threadIdx.x][threadIdx.y] = 0;
        }

        __syncthreads();

#pragma unroll
        for (auto n = 0; n < BLOCK_SIZE; n++) {
            auto i1 = a_shared[threadIdx.y][n];
            auto i2 = b_shared[threadIdx.x][n];
            /* auto val = lut_operator<uint8_t>(tex, i2, i1);*/
            auto val = mitchell_trunc((int16_t)i2, (int16_t)i1);
            acc += val;
        }
        __syncthreads();
    }

    if (batch < res.size(0) && row < res.size(1) && col < res.size(2)) {
        res[batch][row][col] = acc;
    }
}

void ta_gemm_cuda_launch(at::Tensor a, at::Tensor b, at::Tensor lut, at::Tensor res) {
    // prepare the kernel configuration
    const dim3 blocks((res.size(2) + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (res.size(1) + BLOCK_SIZE - 1) / BLOCK_SIZE, res.size(0));
    const dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Create resource description
    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = lut.data_ptr<int32_t>();
    resDesc.res.linear.sizeInBytes = lut.size(0) * lut.size(1) * sizeof(int32_t);
    resDesc.res.linear.desc = cudaCreateChannelDesc<int32_t>();

    // Create texture description
    struct cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    // Create texture
    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

    if (a.dim() == 3) {
        AT_DISPATCH_ALL_TYPES(
            a.scalar_type(), "torchapprox cuda", ([&] {
                ta_gemm_kernel<scalar_t><<<blocks, threads_per_block>>>(
                    tex, a.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    res.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>());
            }));

    } else {
        AT_DISPATCH_ALL_TYPES(
            a.scalar_type(), "torchapprox cuda", ([&] {
                ta_gemm_kernel_batchb<scalar_t><<<blocks, threads_per_block>>>(
                    tex, a.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    b.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    res.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>());
            }));
    }
}
