//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/vec.h"
#include "poly_info_1d.h"
#include "poly_info_2d.h"
#include "poly_info_3d.h"

namespace wp::fields {
template<typename TYPE>
struct grid_data_base_t {
    static constexpr uint32_t dim = TYPE::dim;
    using point_t = vec_t<dim, float>;

    array_t<float> data;

    //! return value of specific point
    CUDA_CALLABLE float value(const point_t &pt, uint32_t idx) {
        return data[idx];
    }
    //! return value of specific point in specific bry
    CUDA_CALLABLE float value(const point_t &pt, uint32_t idx, uint32_t bry) {
        return data[idx];
    }

    //! return value of specific point
    CUDA_CALLABLE fixed_array_t<float, dim> gradient(const point_t &pt, uint32_t idx) {
        return fixed_array_t<float, dim>{};
    }
    //! return value of specific point in specific bry
    CUDA_CALLABLE fixed_array_t<float, dim> gradient(const point_t &pt, uint32_t idx, uint32_t bry) {
        return fixed_array_t<float, dim>{};
    }
};

template<typename TYPE, int order>
struct grid_data_t : public grid_data_base_t<TYPE> {
    static constexpr uint32_t dim = TYPE::dim;
    using point_t = vec_t<dim, float>;

    typename poly_info_t<TYPE, order>::FuncValueFunctor func_value;
    typename poly_info_t<TYPE, order>::FuncGradientFunctor func_gradient;
    array_t<typename poly_info_t<TYPE, order>::Vec> slope;

    CUDA_CALLABLE grid_data_t(const grid_t<TYPE> &grid, poly_info_t<TYPE, order> poly)
        : func_value{grid, poly}, func_gradient{grid} {}

    //! return value of specific point
    CUDA_CALLABLE float value(const point_t &pt, uint32_t idx) {
        return func_value(idx, pt, this->data[idx], slope[idx]);
    }
    //! return value of specific point in specific bry
    CUDA_CALLABLE float value(const point_t &pt, uint32_t idx, uint32_t bry) {
        return func_value(idx, pt, this->data[idx], slope[idx]);
    }

    //! return value of specific point
    CUDA_CALLABLE fixed_array_t<float, dim> gradient(const point_t &pt, uint32_t idx) {
        return func_gradient(idx, pt, slope[idx]);
    }
    //! return value of specific point in specific bry
    CUDA_CALLABLE fixed_array_t<float, dim> gradient(const point_t &pt, uint32_t idx, uint32_t bry) {
        return func_gradient(idx, pt, slope[idx]);
    }
};

}// namespace wp::fields