//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace wp::fields {
template<>
struct base_template_geometry_t<Interval> {
    using base = Interval;
    static constexpr uint32_t dim = Interval::dim;
    using point_t = Interval::point_t;
    static constexpr uint32_t n_point = 2;
    CUDA_CALLABLE static constexpr fixed_array_t<point_t, n_point> points() {
        return {point_t{0},
                point_t{1}};
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

    CUDA_CALLABLE static constexpr auto n_geometry(uint32_t index) {
        if (index == 0) {
            return n_geometry_dim0;
        } else {
            return n_geometry_dim1;
        }
    }
    CUDA_CALLABLE static constexpr auto n_vertex(uint32_t level) {
        if (level == 0) {
            return wp::fields::static_geometry_t<1>::size;
        } else {
            return wp::fields::static_geometry_t<2>::size;
        }
    }
};

template<>
struct template_geometry_t<Interval, 1> : public base_template_geometry_t<Interval> {
    static constexpr uint32_t n_quad_size = 1;
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, n_quad_size>{
            .alg_acc = 1,
            .pnts = {point_t{0}},
            .weights = {1.0}};
    }
};

template<>
struct template_geometry_t<Interval, 2> : public base_template_geometry_t<Interval> {
    static constexpr uint32_t n_quad_size = 2;
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, n_quad_size>{
            .alg_acc = 2,
            .pnts = {point_t{-0.577350269189626},
                     point_t{0.577350269189626}},
            .weights = {0.5, 0.5}};
    }
};

template<>
struct template_geometry_t<Interval, 3> : public base_template_geometry_t<Interval> {
    static constexpr uint32_t n_quad_size = 3;
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, n_quad_size>{
            .alg_acc = 3,
            .pnts = {point_t{-0.774596669241483},
                     point_t{0},
                     point_t{0.774596669241483}},
            .weights = {0.277777777777778, 0.444444444444444, 0.277777777777778}};
    }
};

template<>
struct template_geometry_t<Interval, 4> : public base_template_geometry_t<Interval> {
    static constexpr uint32_t n_quad_size = 4;
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, n_quad_size>{
            .alg_acc = 4,
            .pnts = {point_t{-0.861136311594053},
                     point_t{-0.339981043584856},
                     point_t{0.339981043584856},
                     point_t{0.861136311594053}},
            .weights = {0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727}};
    }
};

template<>
struct template_geometry_t<Interval, 5> : public base_template_geometry_t<Interval> {
    static constexpr uint32_t n_quad_size = 5;
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, n_quad_size>{
            .alg_acc = 5,
            .pnts = {point_t{-0.906179845938664},
                     point_t{-0.538469310105683},
                     point_t{0},
                     point_t{0.538469310105683},
                     point_t{0.906179845938664}},
            .weights = {0.118463442528095, 0.239314335249683, 0.284444444444444, 0.239314335249683, 0.118463442528095}};
    }
};

template<>
struct template_geometry_t<Interval, 6> : public base_template_geometry_t<Interval> {
    static constexpr uint32_t n_quad_size = 6;
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, n_quad_size>{
            .alg_acc = 6,
            .pnts = {point_t{-0.932469514203152},
                     point_t{-0.661209386466264},
                     point_t{-0.238619186083197},
                     point_t{0.238619186083197},
                     point_t{0.661209386466264},
                     point_t{0.932469514203152}},
            .weights = {0.0856622461895852, 0.180380786524069, 0.233956967286346,
                        0.233956967286346, 0.180380786524069, 0.0856622461895852}};
    }
};

template<>
struct template_geometry_t<Interval, 7> : public base_template_geometry_t<Interval> {
    static constexpr uint32_t n_quad_size = 7;
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, n_quad_size>{
            .alg_acc = 7,
            .pnts = {point_t{-0.949107912342758},
                     point_t{-0.741531185599394},
                     point_t{-0.405845151377397},
                     point_t{0},
                     point_t{0.405845151377397},
                     point_t{0.741531185599394},
                     point_t{0.949107912342758}},
            .weights = {0.0647424830844349, 0.139852695744638, 0.190915025252559,
                        0.208979591836735, 0.190915025252559, 0.139852695744638, 0.0647424830844349}};
    }
};

template<>
struct template_geometry_t<Interval, 8> : public base_template_geometry_t<Interval> {
    static constexpr uint32_t n_quad_size = 8;
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, n_quad_size>{
            .alg_acc = 8,
            .pnts = {point_t{-0.960289856497536},
                     point_t{-0.796666477413627},
                     point_t{-0.525532409916329},
                     point_t{-0.18343464249565},
                     point_t{0.18343464249565},
                     point_t{0.525532409916329},
                     point_t{0.796666477413627},
                     point_t{0.960289856497536}},
            .weights = {0.0506142681451881, 0.111190517226687, 0.156853322938944, 0.181341891689181,
                        0.181341891689181, 0.156853322938944, 0.111190517226687, 0.0506142681451881}};
    }
};

template<>
struct template_geometry_t<Interval, 9> : public base_template_geometry_t<Interval> {
    static constexpr uint32_t n_quad_size = 9;
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, n_quad_size>{
            .alg_acc = 9,
            .pnts = {point_t{-0.968160239507626},
                     point_t{-0.836031107326636},
                     point_t{-0.61337143270059},
                     point_t{-0.324253423403809},
                     point_t{0},
                     point_t{0.324253423403809},
                     point_t{0.61337143270059},
                     point_t{0.836031107326636},
                     point_t{0.968160239507626}},
            .weights = {0.0406371941807872, 0.0903240803474287, 0.130305348201468, 0.156173538520001, 0.16511967750063,
                        0.156173538520001, 0.130305348201468, 0.0903240803474287, 0.0406371941807872}};
    }
};

template<>
struct template_geometry_t<Interval, 10> : public base_template_geometry_t<Interval> {
    static constexpr uint32_t n_quad_size = 10;
    CUDA_CALLABLE static constexpr auto quadrature_info() {
        return quadrature_info_t<dim, n_quad_size>{
            .alg_acc = 10,
            .pnts = {point_t{-0.973906528517172},
                     point_t{-0.865063366688985},
                     point_t{-0.679409568299024},
                     point_t{-0.433395394129247},
                     point_t{-0.148874338981631},
                     point_t{0.148874338981631},
                     point_t{0.433395394129247},
                     point_t{0.679409568299024},
                     point_t{0.865063366688985},
                     point_t{0.973906528517172}},
            .weights = {0.0333356721543441, 0.0747256745752903, 0.109543181257991, 0.134633359654998, 0.147762112357376,
                        0.147762112357376, 0.134633359654998, 0.109543181257991, 0.0747256745752903, 0.0333356721543441}};
    }
};

}// namespace wp::fields
