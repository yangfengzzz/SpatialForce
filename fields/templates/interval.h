//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace wp::fields {
template<>
struct base_template_geometry_t<Interval> {
    static constexpr uint32_t dim = Interval::dim;
    using point_t = Interval::point_t;
    static constexpr uint32_t n_point = 2;
    static constexpr point_t pnt[n_point]{point_t{0}, point_t{1}};
    static constexpr auto point(uint32_t index) {
        return pnt[index];
    }

    static constexpr uint32_t n_geometry_dim0 = 2;
    static constexpr static_geometry_t<1> geometry_dim0[n_geometry_dim0]{
        static_geometry_t<1>{
            .ind = 0,
            .vtx = {0},
            .bnd = {0},
        },
        static_geometry_t<1>{
            .ind = 1,
            .vtx = {1},
            .bnd = {1},
        }};
    static constexpr uint32_t n_geometry_dim1 = 1;
    static constexpr static_geometry_t<2> geometry_dim1[n_geometry_dim1]{static_geometry_t<2>{
        .ind = 0,
        .vtx = {0, 1},
        .bnd = {0, 1},
    }};
    static constexpr auto n_geometry(uint32_t index) {
        if (index == 0) {
            return n_geometry_dim0;
        } else {
            return n_geometry_dim1;
        };
    }
    static constexpr auto n_vertex(uint32_t level) {
        if (level == 0) {
            return wp::fields::static_geometry_t<1>::size;
        } else {
            return wp::fields::static_geometry_t<2>::size;
        };
    }
};

template<>
struct template_geometry_t<Interval, 1> : public base_template_geometry_t<Interval> {
    static constexpr uint32_t n_quad_size = 1;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 1,
        .pnts = {point_t{0}},
        .weights = {1.0}};
};
}// namespace wp::fields
