#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#define FULL_WARP_MASK 0xFFFFFFFF

#define CREATE_SHFL_MASK(mask, predicate) unsigned mask = __ballot_sync(FULL_WARP_MASK, (predicate))

static __host__ __device__ __forceinline__ int floor_div(int a, int b) {
    int c = a / b;

    if (c * b > a) {
        c--;
    }

    return c;
}

__device__ inline unsigned get_lane_id() {
    unsigned int lane_id;

#if __clang__
    return __nvvm_read_ptx_sreg_laneid();
#else
    asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
#endif

    return lane_id;
}

enum DepthwiseConv2dDirection { DIRECTION_FORWARD, DIRECTION_BACKWARD };

struct DWConv2dKernelParams {
    int batch;
    int in_h;
    int in_w;
    int in_channel;

    int kernel_h;
    int kernel_w;

    int up_x;
    int up_y;
    int down_x;
    int down_y;

    int pad_x0;
    int pad_x1;
    int pad_y0;
    int pad_y1;

    int out_h;
    int out_w;
    int out_channel;

    int loop_major;
    int n_out;
};

template <typename scalar_t, DepthwiseConv2dDirection direction, int up_x, int up_y, int down_x,
          int down_y, int kernel_h, int kernel_w, int tile_out_h, int tile_out_w>
__global__ void dwconv2d_kernel(int32_t *out, const scalar_t *input, const scalar_t *kernel,
                                const cudaTextureObject_t lut_tex, const DWConv2dKernelParams p) {
    const int tile_in_h = ((tile_out_h - 1) * down_y + kernel_h - 1) / up_y + 1;
    const int tile_in_w = ((tile_out_w - 1) * down_x + kernel_w - 1) / up_x + 1;

    __shared__ scalar_t sk[kernel_h][kernel_w];
    __shared__ scalar_t sx[tile_in_h][tile_in_w];

    int minor_idx = blockIdx.x;
    int tile_out_y = minor_idx;
    minor_idx -= tile_out_y;
    tile_out_y *= tile_out_h;
    int tile_out_x_base = blockIdx.y * tile_out_w;
    int major_idx_base = blockIdx.z * p.loop_major;

    const int major_dim = p.batch * p.in_channel;

    if (tile_out_x_base >= p.out_w | tile_out_y >= p.out_h | major_idx_base >= major_dim) {
        return;
    }

    for (int loop_major = 0, major_idx = major_idx_base;
         loop_major < p.loop_major & major_idx < major_dim; loop_major++, major_idx++) {
        int channel_idx = major_idx % p.in_channel;

        for (int tap_idx = threadIdx.x; tap_idx < kernel_h * kernel_w; tap_idx += blockDim.x) {
            int ky = tap_idx / kernel_w;
            int kx = tap_idx - ky * kernel_w;
            scalar_t v = 0;

            if (kx < p.kernel_w & ky < p.kernel_h) {
                if (direction == DIRECTION_FORWARD) {
                    v = kernel[channel_idx * p.kernel_w * p.kernel_h + ky * p.kernel_w + kx];
                } else {
                    v = kernel[channel_idx * p.kernel_w * p.kernel_h +
                               (p.kernel_h - 1 - ky) * p.kernel_w + (p.kernel_w - 1 - kx)];
                }
            }

            sk[ky][kx] = v;
        }

        __syncthreads();

        for (int loop_x = 0, tile_out_x = tile_out_x_base; loop_x < 1 & tile_out_x < p.out_w;
             loop_x++, tile_out_x += tile_out_w) {
            int tile_mid_x = tile_out_x * down_x + up_x - 1 - p.pad_x0;
            int tile_mid_y = tile_out_y * down_y + up_y - 1 - p.pad_y0;
            int tile_in_x = floor_div(tile_mid_x, up_x);
            int tile_in_y = floor_div(tile_mid_y, up_y);

            for (int in_idx = threadIdx.x; in_idx < tile_in_h * tile_in_w; in_idx += blockDim.x) {
                int rel_in_y = in_idx / tile_in_w;
                int rel_in_x = in_idx - rel_in_y * tile_in_w;
                int in_x = rel_in_x + tile_in_x;
                int in_y = rel_in_y + tile_in_y;

                scalar_t v = 0;

                if (in_x >= 0 & in_y >= 0 & in_x < p.in_w & in_y < p.in_h) {
                    v = input[((major_idx * p.in_h + in_y) * p.in_w + in_x) + minor_idx];
                }

                sx[rel_in_y][rel_in_x] = v;
            }

            __syncthreads();

            for (int out_idx = threadIdx.x; out_idx < tile_out_h * tile_out_w;
                 out_idx += blockDim.x) {
                int rel_out_y = out_idx / tile_out_w;
                int rel_out_x = out_idx - rel_out_y * tile_out_w;
                int out_x = rel_out_x + tile_out_x;
                int out_y = rel_out_y + tile_out_y;

                int mid_x = tile_mid_x + rel_out_x * down_x;
                int mid_y = tile_mid_y + rel_out_y * down_y;
                int in_x = floor_div(mid_x, up_x);
                int in_y = floor_div(mid_y, up_y);
                int rel_in_x = in_x - tile_in_x;
                int rel_in_y = in_y - tile_in_y;
                int kernel_x = (in_x + 1) * up_x - mid_x - 1;
                int kernel_y = (in_y + 1) * up_y - mid_y - 1;

                int32_t v = 0;

#pragma unroll
                for (int y = 0; y < kernel_h / up_y; y++)
#pragma unroll
                    for (int x = 0; x < kernel_w / up_x; x++) {
                        auto i1 = static_cast<uint8_t>(sx[rel_in_y + y][rel_in_x + x]);
                        auto i2 =
                            static_cast<uint8_t>(sk[kernel_y + y * up_y][kernel_x + x * up_x]);

                        auto idx = (i1 << 8) | i2;
                        auto val = tex1Dfetch<int16_t>(lut_tex, idx);
                        v += val;
                    }

                if (out_x < p.out_w & out_y < p.out_h) {
                    out[((major_idx * p.out_h + out_y) * p.out_w + out_x) + minor_idx] = v;
                }
            }
        }
    }
}

template <typename scalar_t, DepthwiseConv2dDirection kDirection, int kBlockSlices,
          bool kEvenHeight, int kFilterHeight, int kFilterWidth>
__global__ void __launch_bounds__(1024, 2)
    dwconv2d_small_kernel(int32_t *out, const scalar_t *input, const scalar_t *kernel,
                          const cudaTextureObject_t lut_tex, const DWConv2dKernelParams p) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_memory[];
    scalar_t *const shared_data = reinterpret_cast<scalar_t *>(shared_memory);

    const int in_height = p.in_h;
    const int in_width = p.in_w;
    const int in_channel = p.in_channel;
    const int filter_height = kFilterHeight > 0 ? kFilterHeight : p.kernel_h;
    const int filter_width = kFilterWidth > 0 ? kFilterWidth : p.kernel_w;
    const int pad_height = p.pad_y0;
    const int pad_width = p.pad_x0;

    const int block_height = blockDim.y;

    const int block_pixels = in_width * block_height;
    const int block_size = block_pixels * kBlockSlices;
    const int in_pixels = in_width * in_height;
    const int in_increment = in_width - 1;
    const int filter_pixels = filter_height * filter_width;
    const int tile_width = in_width + filter_width - 1;
    const int even_height = kEvenHeight || (1 & ~in_height);
    const int tile_height = in_height + filter_height - even_height;
    const int tile_pixels = tile_width * tile_height;
    const int tile_size = tile_pixels * kBlockSlices;
    const int tile_offset = block_height * tile_width;
    const int pad_offset = pad_height * tile_width + pad_width;
    const int in_slices = in_channel * p.batch;
    const int in_blocks = (in_slices + kBlockSlices - 1) / kBlockSlices;

    const int thread_width = threadIdx.x;
    const int thread_height = threadIdx.y;
    const int thread_channel = threadIdx.z;

    const int thread_pix = thread_height * in_width + thread_width;
    const int thread_idx = thread_channel * block_pixels + thread_pix;

    for (int i = thread_idx; i < tile_size; i += block_size) {
        shared_data[i] = scalar_t(0);
    }

    __syncthreads();

    const int tensor_idx = thread_channel * in_pixels + thread_pix;

    const int data_pix = thread_height * tile_width + thread_width;
    const int data_idx = thread_channel * tile_pixels + data_pix;

    const int tile_idx = data_idx + pad_offset;

    const int filter_pix = thread_pix;
    const int filter_channel = thread_channel;
    const int filter_idx = filter_pixels * filter_channel + filter_pix;

    const int max_slice = in_slices - thread_channel;
    const int filter_write_offset = filter_pix < filter_pixels ? tile_size + filter_idx : 0;
    const int filter_read_offset =
        tile_size + (kDirection == DIRECTION_FORWARD ? filter_pixels * filter_channel
                                                     : filter_pixels * (filter_channel + 1));
    const bool skip_second = !kEvenHeight && thread_height + (in_height & 1) == block_height;

    for (int b = blockIdx.x; b < in_blocks; b += gridDim.x) {
        const int slice = b * kBlockSlices;

        const int inout_offset = slice * in_pixels + tensor_idx;
        const bool slice_in_range = slice < max_slice;

        if (slice_in_range) {
            const scalar_t *const in_ptr = inout_offset + input;
            scalar_t *const tile_ptr = tile_idx + shared_data;
            tile_ptr[0] = __ldg(in_ptr);

            if (!skip_second) {
                tile_ptr[tile_offset] = __ldg(block_pixels + in_ptr);
            }
        }

        if (filter_write_offset != 0) {
            const int filter_offset =
                ((slice + filter_channel) % in_channel) * filter_pixels + filter_pix;
            shared_data[filter_write_offset] = __ldg(filter_offset + kernel);
        }

        __syncthreads();

        if (slice_in_range) {
            int32_t sum1 = 0;
            int32_t sum2 = 0;
            int shared_offset = data_idx;
            const scalar_t *filter_ptr = filter_read_offset + shared_data;

#pragma unroll
            for (int r = 0; r < filter_height; ++r) {
#pragma unroll
                for (int c = 0; c < filter_width; ++c) {
                    if (kDirection == DIRECTION_BACKWARD) {
                        filter_ptr--;
                    }

                    const scalar_t filter_value = *filter_ptr;
                    const scalar_t *const tile_ptr = shared_offset + shared_data;

                    auto i1 = static_cast<uint8_t>(filter_value);
                    auto i2 = static_cast<uint8_t>(tile_ptr[0]);
                    auto i3 = static_cast<uint8_t>(tile_ptr[tile_offset]);

                    auto idx1 = (i1 << 8) | i2;
                    auto idx2 = (i1 << 8) | i3;

                    sum1 += tex1Dfetch<int16_t>(lut_tex, idx1);
                    sum2 += tex1Dfetch<int16_t>(lut_tex, idx2);

                    ++shared_offset;

                    if (kDirection == DIRECTION_FORWARD) {
                        filter_ptr++;
                    }
                }

                shared_offset += in_increment;
            }

            int32_t *const out_ptr = inout_offset + out;

            out_ptr[0] = sum1;

            if (!skip_second) {
                out_ptr[block_pixels] = sum2;
            }
        }

        __syncthreads();
    }
}

DWConv2dKernelParams make_conv2d_params(const torch::Tensor &input, const torch::Tensor &kernel,
                                        int up_h, int up_w, int down_h, int down_w, int pad_h0,
                                        int pad_h1, int pad_w0, int pad_w1) {
    DWConv2dKernelParams p;

    p.batch = input.size(0);
    p.in_channel = input.size(1);
    p.in_h = input.size(2);
    p.in_w = input.size(3);
    p.kernel_h = kernel.size(2);
    p.kernel_w = kernel.size(3);
    p.up_x = up_w;
    p.up_y = up_h;
    p.down_x = down_w;
    p.down_y = down_h;
    p.pad_x0 = pad_w0;
    p.pad_x1 = pad_w1;
    p.pad_y0 = pad_h0;
    p.pad_y1 = pad_h1;

    p.out_h = (p.in_h * p.up_y + p.pad_y0 + p.pad_y1 - p.kernel_h + p.down_y) / p.down_y;
    p.out_w = (p.in_w * p.up_x + p.pad_x0 + p.pad_x1 - p.kernel_w + p.down_x) / p.down_x;
    p.out_channel = p.in_channel;
    p.n_out = p.batch * p.in_channel * p.out_h * p.out_w;

    return p;
}

bool use_dwconv2d_small(const torch::Tensor &input, const torch::Tensor &kernel, int up_h, int up_w,
                        int down_h, int down_w, int pad_h, int pad_w) {
    DWConv2dKernelParams p =
        make_conv2d_params(input, kernel, up_h, up_w, down_h, down_w, pad_h, pad_h, pad_w, pad_w);

    return p.down_y == 1 && p.down_x == 1 && p.in_h <= 32 && p.in_w <= 32 && p.in_h == p.out_h &&
           p.in_w == p.out_w && p.pad_y0 >= 0 && p.pad_y0 < p.kernel_h && p.pad_x0 >= 0 &&
           p.pad_x0 < p.kernel_w && p.kernel_h * p.kernel_w <= (p.in_h + 1) / 2 * p.in_w;
}

template <typename scalar_t, DepthwiseConv2dDirection kDirection, int kBlockSlices,
          bool kEvenHeight>
torch::Tensor ta_dwconv2d_small_launch(const torch::Tensor &input, const torch::Tensor &kernel,
                                       const cudaTextureObject_t &lut_tex,
                                       const DWConv2dKernelParams p) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    const int block_height = (p.in_h + 1) / 2;
    dim3 block_dim = dim3(p.in_w, block_height, kBlockSlices);

    const int tile_width = p.in_w + p.kernel_w - 1;
    const int tile_height = block_height * 2 + p.kernel_h - 1;
    const int tile_pixels = tile_height * tile_width;
    const int filter_pixels = p.kernel_h * p.kernel_w;
    const int num_outputs = p.batch * p.out_h * p.out_w * p.out_channel;
    int block_count = std::min(num_outputs / (block_dim.x * block_dim.y * block_dim.z),
                               static_cast<unsigned int>(65535));

    auto options = input.options().dtype(torch::kI32);
    auto out = at::empty({p.batch, p.in_channel, p.out_h, p.out_w}, options);

    const int shared_memory_size = kBlockSlices * (tile_pixels + filter_pixels) * sizeof(scalar_t);

    if (p.kernel_h == 3 && p.kernel_w == 3) {
        dwconv2d_small_kernel<scalar_t, kDirection, kBlockSlices, kEvenHeight, 3, 3>
            <<<block_count, block_dim, shared_memory_size, stream>>>(
                out.data_ptr<int32_t>(), input.data_ptr<scalar_t>(), kernel.data_ptr<scalar_t>(),
                lut_tex, p);
    } else {
        dwconv2d_small_kernel<scalar_t, kDirection, kBlockSlices, kEvenHeight, -1, -1>
            <<<block_count, block_dim, shared_memory_size, stream>>>(
                out.data_ptr<int32_t>(), input.data_ptr<scalar_t>(), kernel.data_ptr<scalar_t>(),
                lut_tex, p);
    }

    return out;
}

template <typename scalar_t, DepthwiseConv2dDirection kDirection, int kBlockSlices>
torch::Tensor ta_dwconv2d_small_launch(const torch::Tensor &input, const torch::Tensor &kernel,
                                       const cudaTextureObject_t &lut_tex,
                                       const DWConv2dKernelParams p) {
    torch::Tensor out;

    if (p.in_h & 1) {
        out = ta_dwconv2d_small_launch<scalar_t, kDirection, kBlockSlices, false>(input, kernel,
                                                                                  lut_tex, p);
    } else {
        out = ta_dwconv2d_small_launch<scalar_t, kDirection, kBlockSlices, true>(input, kernel,
                                                                                 lut_tex, p);
    }

    return out;
}

torch::Tensor ta_dwconv2d_small_launch(const torch::Tensor &input, const torch::Tensor &kernel,
                                       const torch::Tensor &lut, int up_h, int up_w, int down_h,
                                       int down_w, int pad_h, int pad_w, bool forward) {
    DWConv2dKernelParams p =
        make_conv2d_params(input, kernel, up_h, up_w, down_h, down_w, pad_h, pad_h, pad_w, pad_w);

    auto x = input.contiguous();
    auto k = kernel.contiguous();

    p.n_out = p.batch * p.in_channel * p.out_h * p.out_w;

    const int block_pixels = (p.in_h + 1) / 2 * p.in_w;
    torch::Tensor out;

    // Create resource description
    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = lut.data_ptr<int16_t>();
    resDesc.res.linear.sizeInBytes = lut.size(0) * lut.size(1) * sizeof(int16_t);
    resDesc.res.linear.desc = cudaCreateChannelDesc<int16_t>();

    // Create texture description
    struct cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    // Create texture
    cudaTextureObject_t lut_tex;
    cudaCreateTextureObject(&lut_tex, &resDesc, &texDesc, NULL);

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "dwconv2d_small", [&] {
        if (forward) {
            if (block_pixels > 256) {
                out = ta_dwconv2d_small_launch<scalar_t, DIRECTION_FORWARD, 2>(x, k, lut_tex, p);
            } else if (block_pixels > 128) {
                out = ta_dwconv2d_small_launch<scalar_t, DIRECTION_FORWARD, 4>(x, k, lut_tex, p);
            } else {
                out = ta_dwconv2d_small_launch<scalar_t, DIRECTION_FORWARD, 8>(x, k, lut_tex, p);
            }
        } else {
            if (block_pixels > 256) {
                out = ta_dwconv2d_small_launch<scalar_t, DIRECTION_BACKWARD, 2>(x, k, lut_tex, p);
            } else if (block_pixels > 128) {
                out = ta_dwconv2d_small_launch<scalar_t, DIRECTION_BACKWARD, 4>(x, k, lut_tex, p);
            } else {
                out = ta_dwconv2d_small_launch<scalar_t, DIRECTION_BACKWARD, 8>(x, k, lut_tex, p);
            }
        }
    });

    return out;
}

template <typename scalar_t, DepthwiseConv2dDirection direction, int up_x, int up_y, int down_x,
          int down_y, int kernel_h, int kernel_w, int tile_out_h, int tile_out_w>
torch::Tensor ta_dwconv2d_launch(const torch::Tensor &input, const torch::Tensor &kernel,
                                 const cudaTextureObject_t &lut_tex, DWConv2dKernelParams p) {
    int cur_device = -1;
    cudaGetDevice(&cur_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(cur_device);

    auto options = input.options().dtype(torch::kI32);
    auto out = at::empty({p.batch, p.in_channel, p.out_h, p.out_w}, options);

    dim3 block_size;
    dim3 grid_size;

    int major_dim = p.batch * p.in_channel;

    if (tile_out_h > 0 && tile_out_w > 0) {
        p.loop_major = (major_dim - 1) / 16384 + 1;
        block_size = dim3(32 * 8, 1, 1);
        grid_size = dim3(((p.out_h - 1) / tile_out_h + 1), (p.out_w - 1) / tile_out_w + 1,
                         (major_dim - 1) / p.loop_major + 1);
    }

    dwconv2d_kernel<scalar_t, direction, up_x, up_y, down_x, down_y, kernel_h, kernel_w, tile_out_h,
                    tile_out_w>
        <<<grid_size, block_size, 0, stream>>>(out.data_ptr<int32_t>(), input.data_ptr<scalar_t>(),
                                               kernel.data_ptr<scalar_t>(), lut_tex, p);

    return out;
}

template <typename scalar_t, DepthwiseConv2dDirection direction>
torch::Tensor ta_dwconv2d_launch(const torch::Tensor &input, const torch::Tensor &kernel,
                                 const cudaTextureObject_t &lut_tex, DWConv2dKernelParams p) {
    if (p.up_x == 1 && p.up_y == 1 && p.down_x == 1 && p.down_y == 1) {
        if (p.kernel_h <= 3 && p.kernel_w <= 3) {
            return ta_dwconv2d_launch<scalar_t, direction, 1, 1, 1, 1, 3, 3, 16, 64>(input, kernel,
                                                                                     lut_tex, p);

        } else if (p.kernel_h <= 5 && p.kernel_w <= 5) {
            return ta_dwconv2d_launch<scalar_t, direction, 1, 1, 1, 1, 5, 5, 16, 64>(input, kernel,
                                                                                     lut_tex, p);
        } else if (p.kernel_h <= 7 && p.kernel_w <= 7) {
            return ta_dwconv2d_launch<scalar_t, direction, 1, 1, 1, 1, 7, 7, 16, 64>(input, kernel,
                                                                                     lut_tex, p);
        }
    } else if (p.up_x == 2 && p.up_y == 2) {
        if (p.kernel_h <= 4 && p.kernel_w <= 4) {
            return ta_dwconv2d_launch<scalar_t, direction, 2, 2, 1, 1, 4, 4, 16, 64>(input, kernel,
                                                                                     lut_tex, p);
        } else if (p.kernel_h <= 6 && p.kernel_w <= 6) {
            return ta_dwconv2d_launch<scalar_t, direction, 2, 2, 1, 1, 6, 6, 16, 64>(input, kernel,
                                                                                     lut_tex, p);
        } else if (p.kernel_h <= 8 && p.kernel_w <= 8) {
            return ta_dwconv2d_launch<scalar_t, direction, 2, 2, 1, 1, 8, 8, 16, 64>(input, kernel,
                                                                                     lut_tex, p);
        }
    } else if (p.down_x == 2 && p.down_y == 2) {
        if (p.kernel_h <= 4 && p.kernel_w <= 4) {
            return ta_dwconv2d_launch<scalar_t, direction, 1, 1, 2, 2, 4, 4, 8, 32>(input, kernel,
                                                                                    lut_tex, p);
        } else if (p.kernel_h <= 6 && p.kernel_w <= 6) {
            return ta_dwconv2d_launch<scalar_t, direction, 1, 1, 2, 2, 6, 6, 8, 32>(input, kernel,
                                                                                    lut_tex, p);
        } else if (p.kernel_h <= 8 && p.kernel_w <= 8) {
            return ta_dwconv2d_launch<scalar_t, direction, 1, 1, 2, 2, 8, 8, 8, 32>(input, kernel,
                                                                                    lut_tex, p);
        }
    }
    return torch::empty(1);
}

torch::Tensor ta_dwconv2d_launch(const torch::Tensor &input, const torch::Tensor &kernel,
                                 const torch::Tensor &lut, int up_h, int up_w, int down_h,
                                 int down_w, int pad_h0, int pad_h1, int pad_w0, int pad_w1,
                                 bool forward) {
    DWConv2dKernelParams p = make_conv2d_params(input, kernel, up_h, up_w, down_h, down_w, pad_h0,
                                                pad_h1, pad_w0, pad_w1);

    auto x = input.contiguous();
    auto k = kernel.contiguous();

    torch::Tensor out;

    // Create resource description
    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = lut.data_ptr<int16_t>();
    resDesc.res.linear.sizeInBytes = lut.size(0) * lut.size(1) * sizeof(int16_t);
    resDesc.res.linear.desc = cudaCreateChannelDesc<int16_t>();

    // Create texture description
    struct cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    // Create texture
    cudaTextureObject_t lut_tex;
    cudaCreateTextureObject(&lut_tex, &resDesc, &texDesc, NULL);

    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "dwconv2d", [&] {
        if (forward) {
            out = ta_dwconv2d_launch<scalar_t, DIRECTION_FORWARD>(x, k, lut_tex, p);
        } else {
            out = ta_dwconv2d_launch<scalar_t, DIRECTION_BACKWARD>(x, k, lut_tex, p);
        }
    });

    return out;
}
