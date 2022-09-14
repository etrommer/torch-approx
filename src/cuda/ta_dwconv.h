#pragma once
#include <torch/extension.h>

bool use_dwconv2d_small(const torch::Tensor &input, const torch::Tensor &kernel, int up_h, int up_w,
                        int down_h, int down_w, int pad_h, int pad_w);

torch::Tensor ta_dwconv2d_launch(const torch::Tensor &input, const torch::Tensor &kernel, int up_h,
                                 int up_w, int down_h, int down_w, int pad_h0, int pad_h1,
                                 int pad_w0, int pad_w1, bool forward);

torch::Tensor ta_dwconv2d_small_launch(const torch::Tensor &input, const torch::Tensor &kernel,
                                       int up_h, int up_w, int down_h, int down_w, int pad_h,
                                       int pad_w, bool forward);
