//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace wp::fields {
template<>
struct base_template_geometry_t<Tetrahedron> {
    using base = Tetrahedron;
    static constexpr uint32_t dim = Tetrahedron::dim;
    using point_t = Tetrahedron::point_t;
    static constexpr uint32_t n_point = 4;
    static constexpr point_t pnt[n_point]{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    static constexpr auto point(uint32_t index) {
        return pnt[index];
    }

    static constexpr uint32_t n_geometry_dim0 = 4;
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
        },
        static_geometry_t<1>{
            .ind = 2,
            .vtx = {2},
            .bnd = {2},
        },
        static_geometry_t<1>{
            .ind = 3,
            .vtx = {3},
            .bnd = {3},
        }};

    static constexpr uint32_t n_geometry_dim1 = 6;
    static constexpr static_geometry_t<2> geometry_dim1[n_geometry_dim1]{
        static_geometry_t<2>{
            .ind = 0,
            .vtx = {0, 1},
            .bnd = {0, 1},
        },
        static_geometry_t<2>{
            .ind = 1,
            .vtx = {0, 2},
            .bnd = {0, 2},
        },
        static_geometry_t<2>{
            .ind = 2,
            .vtx = {0, 3},
            .bnd = {0, 3},
        },
        static_geometry_t<2>{
            .ind = 3,
            .vtx = {2, 3},
            .bnd = {2, 3},
        },
        static_geometry_t<2>{
            .ind = 4,
            .vtx = {1, 3},
            .bnd = {1, 3},
        },
        static_geometry_t<2>{
            .ind = 5,
            .vtx = {1, 2},
            .bnd = {1, 2},
        },
    };

    static constexpr uint32_t n_geometry_dim2 = 4;
    static constexpr static_geometry_t<3> geometry_dim2[n_geometry_dim2]{
        static_geometry_t<3>{
            .ind = 0,
            .vtx = {1, 2, 3},
            .bnd = {3, 4, 5},
        },
        static_geometry_t<3>{
            .ind = 1,
            .vtx = {0, 2, 3},
            .bnd = {3, 2, 1},
        },
        static_geometry_t<3>{
            .ind = 2,
            .vtx = {0, 1, 3},
            .bnd = {4, 2, 0},
        },
        static_geometry_t<3>{
            .ind = 3,
            .vtx = {0, 1, 2},
            .bnd = {5, 1, 0},
        },
    };

    static constexpr uint32_t n_geometry_dim3 = 1;
    static constexpr static_geometry_t<4> geometry_dim3[n_geometry_dim3]{
        static_geometry_t<4>{
            .ind = 0,
            .vtx = {0, 1, 2, 3},
            .bnd = {0, 1, 2, 3},
        },
    };

    static constexpr auto n_geometry(uint32_t index) {
        if (index == 0) {
            return n_geometry_dim0;
        } else if (index == 1) {
            return n_geometry_dim1;
        } else if (index == 2) {
            return n_geometry_dim2;
        } else {
            return n_geometry_dim3;
        };
    }
    static constexpr auto n_vertex(uint32_t level) {
        if (level == 0) {
            return wp::fields::static_geometry_t<1>::size;
        } else if (level == 1) {
            return wp::fields::static_geometry_t<2>::size;
        } else if (level == 2) {
            return wp::fields::static_geometry_t<3>::size;
        } else {
            return wp::fields::static_geometry_t<4>::size;
        };
    }
};

template<>
struct template_geometry_t<Tetrahedron, 1> : public base_template_geometry_t<Tetrahedron> {
    static constexpr uint32_t n_quad_size = 1;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 1,
        .pnts = {{0.25000000, 0.25000000, 0.25000000}},
        .weights = {1.0}};
};

template<>
struct template_geometry_t<Tetrahedron, 2> : public base_template_geometry_t<Tetrahedron> {
    static constexpr uint32_t n_quad_size = 4;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 2,
        .pnts = {{0.13819660, 0.13819660, 0.13819660},
                 {0.58541020, 0.13819660, 0.13819660},
                 {0.13819660, 0.58541020, 0.13819660},
                 {0.13819660, 0.13819660, 0.58541020}},
        .weights = {0.25, 0.25, 0.25, 0.25}};
};

template<>
struct template_geometry_t<Tetrahedron, 3> : public base_template_geometry_t<Tetrahedron> {
    static constexpr uint32_t n_quad_size = 5;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 3,
        .pnts = {
            {0.25000000, 0.25000000, 0.25000000},
            {0.16666667, 0.16666667, 0.16666667},
            {0.50000000, 0.16666667, 0.16666667},
            {0.16666667, 0.50000000, 0.16666667},
            {0.16666667, 0.16666667, 0.50000000},
        },
        .weights = {-0.8, 0.45, 0.45, 0.45, 0.45}};
};

template<>
struct template_geometry_t<Tetrahedron, 4> : public base_template_geometry_t<Tetrahedron> {
    static constexpr uint32_t n_quad_size = 11;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 4,
        .pnts = {
            {0.2500000000000000, 0.2500000000000000, 0.2500000000000000},
            {0.7857142857142857, 0.0714285714285714, 0.0714285714285714},
            {0.0714285714285714, 0.0714285714285714, 0.0714285714285714},
            {0.0714285714285714, 0.0714285714285714, 0.7857142857142857},
            {0.0714285714285714, 0.7857142857142857, 0.0714285714285714},
            {0.1005964238332008, 0.3994035761667992, 0.3994035761667992},
            {0.3994035761667992, 0.1005964238332008, 0.3994035761667992},
            {0.3994035761667992, 0.3994035761667992, 0.1005964238332008},
            {0.3994035761667992, 0.1005964238332008, 0.1005964238332008},
            {0.1005964238332008, 0.3994035761667992, 0.1005964238332008},
            {0.1005964238332008, 0.1005964238332008, 0.3994035761667992},
        },
        .weights = {-0.0789333333333333, 0.0457333333333333, 0.0457333333333333,//
                    0.0457333333333333, 0.0457333333333333, 0.1493333333333333, //
                    0.1493333333333333, 0.1493333333333333, 0.1493333333333333, //
                    0.1493333333333333, 0.1493333333333333}};
};

template<>
struct template_geometry_t<Tetrahedron, 6> : public base_template_geometry_t<Tetrahedron> {
    static constexpr uint32_t n_quad_size = 24;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 6,
        .pnts = {
            {0.3561913862225449, 0.2146028712591517, 0.2146028712591517},
            {0.2146028712591517, 0.2146028712591517, 0.2146028712591517},
            {0.2146028712591517, 0.2146028712591517, 0.3561913862225449},
            {0.2146028712591517, 0.3561913862225449, 0.2146028712591517},
            {0.8779781243961660, 0.0406739585346113, 0.0406739585346113},
            {0.0406739585346113, 0.0406739585346113, 0.0406739585346113},
            {0.0406739585346113, 0.0406739585346113, 0.8779781243961660},
            {0.0406739585346113, 0.8779781243961660, 0.0406739585346113},
            {0.0329863295731731, 0.3223378901422757, 0.3223378901422757},
            {0.3223378901422757, 0.3223378901422757, 0.3223378901422757},
            {0.3223378901422757, 0.3223378901422757, 0.0329863295731731},
            {0.3223378901422757, 0.0329863295731731, 0.3223378901422757},
            {0.2696723314583159, 0.0636610018750175, 0.0636610018750175},
            {0.0636610018750175, 0.2696723314583159, 0.0636610018750175},
            {0.0636610018750175, 0.0636610018750175, 0.2696723314583159},
            {0.6030056647916491, 0.0636610018750175, 0.0636610018750175},
            {0.0636610018750175, 0.6030056647916491, 0.0636610018750175},
            {0.0636610018750175, 0.0636610018750175, 0.6030056647916491},
            {0.0636610018750175, 0.2696723314583159, 0.6030056647916491},
            {0.2696723314583159, 0.6030056647916491, 0.0636610018750175},
            {0.6030056647916491, 0.0636610018750175, 0.2696723314583159},
            {0.0636610018750175, 0.6030056647916491, 0.2696723314583159},
            {0.2696723314583159, 0.0636610018750175, 0.6030056647916491},
            {0.6030056647916491, 0.2696723314583159, 0.0636610018750175},
        },
        .weights = {0.0399227502581679, 0.0399227502581679, 0.0399227502581679,//
                    0.0399227502581679, 0.0100772110553207, 0.0100772110553207,//
                    0.0100772110553207, 0.0100772110553207, 0.0553571815436544,//
                    0.0553571815436544, 0.0553571815436544, 0.0553571815436544,//
                    0.0482142857142857, 0.0482142857142857, 0.0482142857142857,//
                    0.0482142857142857, 0.0482142857142857, 0.0482142857142857,//
                    0.0482142857142857, 0.0482142857142857, 0.0482142857142857,//
                    0.0482142857142857, 0.0482142857142857, 0.0482142857142857}};
};

template<>
struct template_geometry_t<Tetrahedron, 8> : public base_template_geometry_t<Tetrahedron> {
    static constexpr uint32_t n_quad_size = 45;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 8,
        .pnts = {
            {0.2500000000000000, 0.2500000000000000, 0.2500000000000000},
            {0.6175871903000830, 0.1274709365666390, 0.1274709365666390},
            {0.1274709365666390, 0.1274709365666390, 0.1274709365666390},
            {0.1274709365666390, 0.1274709365666390, 0.6175871903000830},
            {0.1274709365666390, 0.6175871903000830, 0.1274709365666390},
            {0.9037635088221031, 0.0320788303926323, 0.0320788303926323},
            {0.0320788303926323, 0.0320788303926323, 0.0320788303926323},
            {0.0320788303926323, 0.0320788303926323, 0.9037635088221031},
            {0.0320788303926323, 0.9037635088221031, 0.0320788303926323},
            {0.4502229043567190, 0.0497770956432810, 0.0497770956432810},
            {0.0497770956432810, 0.4502229043567190, 0.0497770956432810},
            {0.0497770956432810, 0.0497770956432810, 0.4502229043567190},
            {0.0497770956432810, 0.4502229043567190, 0.4502229043567190},
            {0.4502229043567190, 0.0497770956432810, 0.4502229043567190},
            {0.4502229043567190, 0.4502229043567190, 0.0497770956432810},
            {0.3162695526014501, 0.1837304473985499, 0.1837304473985499},
            {0.1837304473985499, 0.3162695526014501, 0.1837304473985499},
            {0.1837304473985499, 0.1837304473985499, 0.3162695526014501},
            {0.1837304473985499, 0.3162695526014501, 0.3162695526014501},
            {0.3162695526014501, 0.1837304473985499, 0.3162695526014501},
            {0.3162695526014501, 0.3162695526014501, 0.1837304473985499},
            {0.0229177878448171, 0.2319010893971509, 0.2319010893971509},
            {0.2319010893971509, 0.0229177878448171, 0.2319010893971509},
            {0.2319010893971509, 0.2319010893971509, 0.0229177878448171},
            {0.5132800333608811, 0.2319010893971509, 0.2319010893971509},
            {0.2319010893971509, 0.5132800333608811, 0.2319010893971509},
            {0.2319010893971509, 0.2319010893971509, 0.5132800333608811},
            {0.2319010893971509, 0.0229177878448171, 0.5132800333608811},
            {0.0229177878448171, 0.5132800333608811, 0.2319010893971509},
            {0.5132800333608811, 0.2319010893971509, 0.0229177878448171},
            {0.2319010893971509, 0.5132800333608811, 0.0229177878448171},
            {0.0229177878448171, 0.2319010893971509, 0.5132800333608811},
            {0.5132800333608811, 0.0229177878448171, 0.2319010893971509},
            {0.7303134278075384, 0.0379700484718286, 0.0379700484718286},
            {0.0379700484718286, 0.7303134278075384, 0.0379700484718286},
            {0.0379700484718286, 0.0379700484718286, 0.7303134278075384},
            {0.1937464752488044, 0.0379700484718286, 0.0379700484718286},
            {0.0379700484718286, 0.1937464752488044, 0.0379700484718286},
            {0.0379700484718286, 0.0379700484718286, 0.1937464752488044},
            {0.0379700484718286, 0.7303134278075384, 0.1937464752488044},
            {0.7303134278075384, 0.1937464752488044, 0.0379700484718286},
            {0.1937464752488044, 0.0379700484718286, 0.7303134278075384},
            {0.0379700484718286, 0.1937464752488044, 0.7303134278075384},
            {0.7303134278075384, 0.0379700484718286, 0.1937464752488044},
            {0.1937464752488044, 0.7303134278075384, 0.0379700484718286},
        },
        .weights = {
            -0.2359620398477557,//
            0.0244878963560562, //
            0.0244878963560562, //
            0.0244878963560562, //
            0.0244878963560562, //
            0.0039485206398261, //
            0.0039485206398261, //
            0.0039485206398261, //
            0.0039485206398261, //
            0.0263055529507371, //
            0.0263055529507371, //
            0.0263055529507371, //
            0.0263055529507371, //
            0.0263055529507371, //
            0.0263055529507371, //
            0.0829803830550589, //
            0.0829803830550589, //
            0.0829803830550589, //
            0.0829803830550589, //
            0.0829803830550589, //
            0.0829803830550589, //
            0.0254426245481023, //
            0.0254426245481023, //
            0.0254426245481023, //
            0.0254426245481023, //
            0.0254426245481023, //
            0.0254426245481023, //
            0.0254426245481023, //
            0.0254426245481023, //
            0.0254426245481023, //
            0.0254426245481023, //
            0.0254426245481023, //
            0.0254426245481023, //
            0.0134324384376852, //
            0.0134324384376852, //
            0.0134324384376852, //
            0.0134324384376852, //
            0.0134324384376852, //
            0.0134324384376852, //
            0.0134324384376852, //
            0.0134324384376852, //
            0.0134324384376852, //
            0.0134324384376852, //
            0.0134324384376852, //
            0.0134324384376852,
        }};
};

}// namespace wp::fields