//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "optimizer.h"

namespace wp {
__global__ void gd_step(array_t<float> arr_x,
                        array_t<float> arr_dfdx,
                        float alpha) {
    auto tid = wp::tid();

    auto x = arr_x[tid];
    auto dfdx = arr_dfdx[tid];

    x = x - dfdx * alpha;

    arr_x[tid] = x;
}

__global__ void nesterov1(float beta, array_t<float> x, array_t<float> x_prev, array_t<float> y) {
    auto tid = wp::tid();

    y[tid] = x[tid] + beta * (x[tid] - x_prev[tid]);
}

__global__ void nesterov2(float alpha, array_t<float> beta, array_t<float> eta,
                          array_t<float> x, array_t<float> x_prev, array_t<float> y,
                          array_t<float> dfdx) {
    auto tid = wp::tid();

    x_prev[tid] = x[tid];
    x[tid] = y[tid] - alpha * dfdx[tid];
}

Optimizer::Optimizer() = default;

void Optimizer::solve(array_t<float> x, const std::function<void()> &grad_func, int max_iters, float alpha, bool report) {
}

}// namespace wp