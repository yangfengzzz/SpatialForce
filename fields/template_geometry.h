//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "geometry.h"
#include "geometry_trait.h"

namespace wp::fields {
template<typename TYPE>
struct base_template_geometry_t {};

template<typename TYPE, uint32_t ACCURACY>
struct template_geometry_t : public base_template_geometry_t<TYPE> {
    static constexpr uint32_t accuracy = ACCURACY;
};

template<uint32_t DIM, uint32_t SIZE>
struct quadrature_info_t {
    static constexpr uint32_t dim = DIM;
    static constexpr uint32_t size = SIZE;
    /// Algebraic accuracy.
    int32_t alg_acc{};
    /// The coordinate of quadrature point.
    fixed_array_t<vec_t<dim, float>, size> pnts;
    /// Quadrature weight on the point.
    fixed_array_t<float, size> weights;
};

}// namespace wp::fields

#include "templates/interval.h"
#include "templates/triangle.h"
#include "templates/tetrahedron.h"