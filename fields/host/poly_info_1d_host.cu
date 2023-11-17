//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "poly_info_1d_host.h"
#include "runtime/cuda_util.h"
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace wp::fields {
namespace {
template<int order>
struct BuildBasisFuncFunctor {
    inline CUDA_CALLABLE BuildBasisFuncFunctor(const grid_t<1> &grid,
                                               array_t<fixed_array_t<float, PolyInfo<1, order>::n_unknown>> poly_constants) {
        size = grid.size;
        output = poly_constants;
    }

    template<typename Index>
    inline CUDA_CALLABLE void operator()(Index ele_idx) {
        for (int m = 1; m <= order; ++m) {
            float J0 = pow(size[ele_idx] / 2.0, m + 1);
            J0 -= pow(-size[ele_idx] / 2.0, m + 1);
            J0 /= float(m + 1);

            J0 /= size[ele_idx];
            output[ele_idx][m - 1] = J0;
        }
    }

private:
    array_t<float> size;
    array_t<fixed_array_t<float, PolyInfo<1, order>::n_unknown>> output;
};
}// namespace

template<int order>
void PolyInfo<1, order>::build_basis_func() {
    thrust::for_each(thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(0) + grid->n_geometry(1),
                     BuildBasisFuncFunctor<order>(grid->grid_handle, handle.poly_constants));
}

template<int order>
void PolyInfo<1, order>::sync_h2d() {
    handle.poly_constants = alloc_array(poly_constants);
}

template<int order>
PolyInfo<1, order>::~PolyInfo() {
    free_array(handle.poly_constants);
}

template class PolyInfo<1, 1>;
template class PolyInfo<1, 2>;
template class PolyInfo<1, 3>;
}// namespace wp::fields