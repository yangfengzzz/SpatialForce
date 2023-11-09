//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/vec.h"

namespace wp::fields {

template<uint32_t dim>
struct quadrature_info_t {
    /// Algebraic accuracy.
    int32_t alg_acc{};
    /// The coordinate of quadrature point.
    array_t<vec_t<dim, float>> pnts;
    /// Quadrature weight on the point.
    array_t<float> weights;
};

template<uint32_t dim>
struct quadrature_info_admin_t {
    array_t<quadrature_info_t<dim>> infos;
    /// Algebraic accuracy table.
    array_t<int32_t> acc_tbl;
};

template<uint32_t DIM, uint32_t SIZE>
struct static_quadrature_info_t {
    static constexpr uint32_t dim = DIM;
    static constexpr uint32_t size = SIZE;
    /// Algebraic accuracy.
    int32_t alg_acc{};
    /// The coordinate of quadrature point.
    vec_t<dim, float> pnts[size];
    /// Quadrature weight on the point.
    float weights[size]{};
};

}// namespace wp::fields
