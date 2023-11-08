//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/vec.h"

namespace wp {
namespace fields {

template <uint32_t DIM>
struct quadrature_info_t {
    /// Algebraic accuracy.
    int32_t alg_acc{};
    /// The coordinate of quadrature point.
    array_t<vec3> pnts;
    /// Quadrature weight on the point.
    array_t<float> weights;
};

template <uint32_t DIM>
struct quadrature_info_admin_t {
    array_t<quadrature_info_t<DIM>> infos;
    /// Algebraic accuracy table.
    array_t<int32_t> acc_tbl;
};

}  // namespace fields
}  // namespace wp