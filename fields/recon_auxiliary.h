//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/mat.h"
#include "poly_info_1d.h"
#include "poly_info_2d.h"
#include "poly_info_3d.h"
#include "grid.h"

namespace wp::fields {
template<typename TYPE, uint32_t ORDER>
struct recon_auxiliary_t {
    array_t<uint32_t> patch_prefix_sum;
    array_t<int32_t> patch;
    array_t<fixed_array_t<float, poly_info_t<TYPE, ORDER>::n_unknown>> patch_polys;
    array_t<typename poly_info_t<TYPE, ORDER>::Mat> G_inv;

    CUDA_CALLABLE uint32_t n_patch(uint32_t ele_idx) {
        if (ele_idx == 0) {
            return patch_prefix_sum[0];
        } else {
            return patch_prefix_sum[ele_idx] - patch_prefix_sum[ele_idx - 1];
        }
    }

    CUDA_CALLABLE array_t<int32_t> get_patch(uint32_t ele_idx) {
        if (ele_idx == 0) {
            return {patch.data, int(patch_prefix_sum[0])};
        } else {
            return {patch.data + patch_prefix_sum[ele_idx - 1], int(patch_prefix_sum[ele_idx] - patch_prefix_sum[ele_idx - 1])};
        }
    }

    CUDA_CALLABLE int32_t *get_patch(uint32_t ele_idx, uint32_t j) {
        if (ele_idx == 0) {
            return patch.data + j;
        } else {
            return patch.data + patch_prefix_sum[ele_idx - 1] + j;
        }
    }

    CUDA_CALLABLE array_t<fixed_array_t<float, poly_info_t<TYPE, ORDER>::n_unknown>> get_poly_avgs(uint32_t ele_idx) {
        if (ele_idx == 0) {
            return {patch_polys.data, int(patch_prefix_sum[0])};
        } else {
            return {patch_polys.data + patch_prefix_sum[ele_idx - 1],
                    int(patch_prefix_sum[ele_idx] - patch_prefix_sum[ele_idx - 1])};
        }
    }

    CUDA_CALLABLE fixed_array_t<float, poly_info_t<TYPE, ORDER>::n_unknown> *get_poly_avgs(uint32_t ele_idx, uint32_t j) {
        if (ele_idx == 0) {
            return patch_polys.data + j;
        } else {
            return patch_polys.data + patch_prefix_sum[ele_idx - 1] + j;
        }
    }

    CUDA_CALLABLE typename poly_info_t<TYPE, ORDER>::Mat *get_g_inv(uint32_t ele_idx) {
        return G_inv.data + ele_idx;
    }
};

}// namespace wp::fields