//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/mat.h"
#include "poly_info.h"
#include "grid.h"

namespace wp::fields {

template<int ORDER>
struct poly_info_t<2, ORDER> {
    static constexpr int dim = 2;
    static constexpr int order = ORDER;
    static constexpr int n_unknown = (order + 2) * (order + 1) / 2 - 1;
    array_t<fixed_array_t<float, n_unknown>> poly_constants;
    using grid_t = grid_t<Triangle>;
    using point_t = grid_t::point_t;
    using Mat = mat_t<n_unknown, n_unknown, float>;
    using Vec = mat_t<n_unknown, 1, float>;
};

}// namespace wp::fields