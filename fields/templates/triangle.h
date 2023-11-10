//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace wp::fields {
template<>
struct base_template_geometry_t<Triangle> {
    static constexpr uint32_t dim = Triangle::dim;
    using point_t = Triangle::point_t;
    static constexpr uint32_t n_point = 3;
    static constexpr point_t pnt[n_point]{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    static constexpr auto point(uint32_t index) {
        return pnt[index];
    }
};

template<>
struct template_geometry_t<Triangle, 1> : public base_template_geometry_t<Triangle> {
    static constexpr uint32_t n_quad_size = 1;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 1,
        .pnts = {{0.3333333333333333, 0.3333333333333333}},
        .weights = {1.0}};
};

}// namespace wp::fields