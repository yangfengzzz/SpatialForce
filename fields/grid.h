//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/vec.h"
#include "core/fixed_array.h"
#include "fields/geometry.h"

namespace wp::fields {
template<uint32_t DIM>
struct grid_t {
    static constexpr uint32_t dim = DIM;
    using point_t = vec_t<dim, float>;

    array_t<point_t> bary_center;
    array_t<float> volume;
    array_t<float> size;

    array_t<fixed_array_t<int32_t, 2>> neighbour;
    array_t<fixed_array_t<int32_t, 2>> period_bry;
    array_t<point_t> boundary_center;
    array_t<float> bry_size;
    array_t<int32_t> boundary_mark;
};
}// namespace wp::fields