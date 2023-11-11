//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace wp::fields {
template<>
struct base_template_geometry_t<Triangle> {
    using base = Triangle;
    static constexpr uint32_t dim = Triangle::dim;
    using point_t = Triangle::point_t;
    static constexpr uint32_t n_point = 3;
    static constexpr point_t pnt[n_point]{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    static constexpr auto point(uint32_t index) {
        return pnt[index];
    }

    static constexpr uint32_t n_geometry_dim0 = 3;
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
        }};

    static constexpr uint32_t n_geometry_dim1 = 3;
    static constexpr static_geometry_t<2> geometry_dim1[n_geometry_dim1]{
        {
            .ind = 0,
            .vtx = {1, 2},
            .bnd = {1, 2},
        },
        {
            .ind = 1,
            .vtx = {0, 2},
            .bnd = {0, 2},
        },
        {
            .ind = 2,
            .vtx = {0, 1},
            .bnd = {0, 1},
        },
    };

    static constexpr uint32_t n_geometry_dim2 = 1;
    static constexpr static_geometry_t<3> geometry_dim2[n_geometry_dim1]{
        {
            .ind = 0,
            .vtx = {0, 1, 2},
            .bnd = {0, 1, 2},
        },
    };

    static constexpr auto n_geometry(uint32_t index) {
        if (index == 0) {
            return n_geometry_dim0;
        } else if (index == 1) {
            return n_geometry_dim1;
        } else {
            return n_geometry_dim2;
        };
    }
    static constexpr auto n_vertex(uint32_t level) {
        if (level == 0) {
            return wp::fields::static_geometry_t<1>::size;
        } else if (level == 1) {
            return wp::fields::static_geometry_t<2>::size;
        } else {
            return wp::fields::static_geometry_t<3>::size;
        };
    }
};

template<>
struct template_geometry_t<Triangle, 1> : public base_template_geometry_t<Triangle> {
    static constexpr uint32_t n_quad_size = 1;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 1,// same as template accuracy
        .pnts = {{0.3333333333333333, 0.3333333333333333}},
        .weights = {1.0}};
};

template<>
struct template_geometry_t<Triangle, 2> : public base_template_geometry_t<Triangle> {
    static constexpr uint32_t n_quad_size = 3;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 2,
        .pnts = {point_t{0.166666666666667, 0.166666666666667},
                 point_t{0.166666666666667, 0.666666666666667},
                 point_t{0.666666666666667, 0.166666666666667}},
        .weights = {0.333333333333333, 0.333333333333333, 0.333333333333333}};
};

template<>
struct template_geometry_t<Triangle, 3> : public base_template_geometry_t<Triangle> {
    static constexpr uint32_t n_quad_size = 4;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 3,
        .pnts = {point_t{0.333333333333333, 0.333333333333333},
                 point_t{0.200000000000000, 0.200000000000000},
                 point_t{0.200000000000000, 0.600000000000000},
                 point_t{0.600000000000000, 0.200000000000000}},
        .weights = {-0.562500000000000, 0.520833333333333, 0.520833333333333, 0.520833333333333}};
};

template<>
struct template_geometry_t<Triangle, 4> : public base_template_geometry_t<Triangle> {
    // http://web.eng.fiu.edu/~advanced-materials/FEA/Gaussian%20Quadrature1.pdf
    static constexpr uint32_t n_quad_size = 6;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 4,
        .pnts = {point_t{0.445948490915965, 0.445948490915965},
                 point_t{0.445948490915965, 0.108103018168070},
                 point_t{0.108103018168070, 0.445948490915965},
                 point_t{0.091576213509771, 0.091576213509771},
                 point_t{0.091576213509771, 0.816847572980459},
                 point_t{0.816847572980459, 0.091576213509771}},
        .weights = {0.223381589678011, 0.223381589678011, 0.223381589678011,//
                    0.109951743655322, 0.109951743655322, 0.109951743655322}};
};

template<>
struct template_geometry_t<Triangle, 5> : public base_template_geometry_t<Triangle> {
    static constexpr uint32_t n_quad_size = 7;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 5,
        .pnts = {point_t{0.333333333333333, 0.333333333333333},
                 point_t{0.470142064105115, 0.470142064105115},
                 point_t{0.470142064105115, 0.059715871789770},
                 point_t{0.059715871789770, 0.470142064105115},
                 point_t{0.101286507323456, 0.101286507323456},
                 point_t{0.101286507323456, 0.797426985353087},
                 point_t{0.797426985353087, 0.101286507323456}},
        .weights = {0.225000000000000, 0.132394152788506, 0.132394152788506, 0.132394152788506,//
                    0.125939180544827, 0.125939180544827, 0.125939180544827}};
};

template<>
struct template_geometry_t<Triangle, 6> : public base_template_geometry_t<Triangle> {
    static constexpr uint32_t n_quad_size = 12;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 6,
        .pnts = {point_t{0.249286745170910, 0.249286745170910},
                 point_t{0.249286745170910, 0.501426509658179},
                 point_t{0.501426509658179, 0.249286745170910},
                 point_t{0.063089014491502, 0.063089014491502},
                 point_t{0.063089014491502, 0.873821971016996},
                 point_t{0.873821971016996, 0.063089014491502},
                 point_t{0.310352451033784, 0.636502499121399},
                 point_t{0.636502499121399, 0.053145049844817},
                 point_t{0.053145049844817, 0.310352451033784},
                 point_t{0.636502499121399, 0.310352451033784},
                 point_t{0.310352451033784, 0.053145049844817},
                 point_t{0.053145049844817, 0.636502499121399}},
        .weights = {0.116786275726379, 0.116786275726379, 0.116786275726379,//
                    0.050844906370207, 0.050844906370207, 0.050844906370207,
                    0.082851075618374, 0.082851075618374, 0.082851075618374,
                    0.082851075618374, 0.082851075618374, 0.082851075618374}};
};

template<>
struct template_geometry_t<Triangle, 7> : public base_template_geometry_t<Triangle> {
    static constexpr uint32_t n_quad_size = 13;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 7,
        .pnts = {point_t{0.333333333333333, 0.333333333333333},
                 point_t{0.260345966079040, 0.260345966079040},
                 point_t{0.260345966079040, 0.479308067841920},
                 point_t{0.479308067841920, 0.260345966079040},
                 point_t{0.065130102902216, 0.065130102902216},
                 point_t{0.065130102902216, 0.869739794195568},
                 point_t{0.869739794195568, 0.065130102902216},
                 point_t{0.312865496004874, 0.638444188569810},
                 point_t{0.638444188569810, 0.048690315425316},
                 point_t{0.048690315425316, 0.312865496004874},
                 point_t{0.638444188569810, 0.312865496004874},
                 point_t{0.312865496004874, 0.048690315425316},
                 point_t{0.048690315425316, 0.638444188569810}},
        .weights = {-0.149570044467682, 0.175615257433208, 0.175615257433208, 0.175615257433208,//
                    0.053347235608838, 0.053347235608838, 0.053347235608838,
                    0.077113760890257, 0.077113760890257, 0.077113760890257,
                    0.077113760890257, 0.077113760890257, 0.077113760890257}};
};

/// Intel. J. Num. Meth. Eng., v21, 1129-1148.
template<>
struct template_geometry_t<Triangle, 8> : public base_template_geometry_t<Triangle> {
    static constexpr uint32_t n_quad_size = 16;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 8,
        .pnts = {point_t{0.333333333333333, 0.333333333333333},
                 point_t{0.081414823414554, 0.459292588292723},
                 point_t{0.658861384496480, 0.170569307751760},
                 point_t{0.898905543365938, 0.050547228317031},
                 point_t{0.008394777409958, 0.263112829634638},
                 point_t{0.459292588292723, 0.459292588292723},
                 point_t{0.170569307751760, 0.170569307751760},
                 point_t{0.050547228317031, 0.050547228317031},
                 point_t{0.263112829634638, 0.728492392955404},
                 point_t{0.459292588292723, 0.081414823414554},
                 point_t{0.170569307751760, 0.658861384496480},
                 point_t{0.050547228317031, 0.898905543365938},
                 point_t{0.728492392955404, 0.008394777409958},
                 point_t{0.263112829634638, 0.008394777409958},
                 point_t{0.728492392955404, 0.263112829634638},
                 point_t{0.008394777409958, 0.728492392955404}},
        .weights = {0.144315607677787, 0.095091634267285, 0.103217370534718, 0.032458497623198,//
                    0.027230314174435, 0.095091634267285, 0.103217370534718, 0.032458497623198,
                    0.027230314174435, 0.095091634267285, 0.103217370534718, 0.032458497623198,
                    0.027230314174435, 0.027230314174435, 0.027230314174435, 0.027230314174435}};
};

template<>
struct template_geometry_t<Triangle, 9> : public base_template_geometry_t<Triangle> {
    static constexpr uint32_t n_quad_size = 19;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 9,
        .pnts = {point_t{0.333333333333333, 0.333333333333333},
                 point_t{0.020634961602525, 0.489682519198738},
                 point_t{0.125820817014127, 0.437089591492937},
                 point_t{0.623592928761935, 0.188203535619033},
                 point_t{0.910540973211095, 0.044729513394453},
                 point_t{0.036838412054736, 0.221962989160766},
                 point_t{0.489682519198738, 0.489682519198738},
                 point_t{0.437089591492937, 0.437089591492937},
                 point_t{0.188203535619033, 0.188203535619033},
                 point_t{0.044729513394453, 0.044729513394453},
                 point_t{0.221962989160766, 0.741198598784498},
                 point_t{0.489682519198738, 0.020634961602525},
                 point_t{0.437089591492937, 0.125820817014127},
                 point_t{0.188203535619033, 0.623592928761935},
                 point_t{0.044729513394453, 0.910540973211095},
                 point_t{0.741198598784498, 0.036838412054736},
                 point_t{0.221962989160766, 0.036838412054736},
                 point_t{0.741198598784498, 0.221962989160766},
                 point_t{0.036838412054736, 0.741198598784498}},
        .weights = {0.097135796282799,
                    0.031334700227139,
                    0.077827541004774,
                    0.079647738927210,
                    0.025577675658698,
                    0.043283539377289,
                    0.031334700227139,
                    0.077827541004774,
                    0.079647738927210,
                    0.025577675658698,
                    0.043283539377289,
                    0.031334700227139,
                    0.077827541004774,
                    0.079647738927210,
                    0.025577675658698,
                    0.043283539377289,
                    0.043283539377289,
                    0.043283539377289,
                    0.043283539377289}};
};

template<>
struct template_geometry_t<Triangle, 10> : public base_template_geometry_t<Triangle> {
    static constexpr uint32_t n_quad_size = 25;
    static constexpr quadrature_info_t<dim, n_quad_size> quadrature_info{
        .alg_acc = 10,
        .pnts = {
            point_t{0.333333333333333, 0.333333333333333},
            point_t{0.028844733232685, 0.485577633383657},
            point_t{0.485577633383657, 0.485577633383657},
            point_t{0.485577633383657, 0.028844733232685},
            point_t{0.781036849029926, 0.109481575485037},
            point_t{0.109481575485037, 0.109481575485037},
            point_t{0.109481575485037, 0.781036849029926},
            point_t{0.141707219414880, 0.307939838764121},
            point_t{0.141707219414880, 0.550352941820999},
            point_t{0.307939838764121, 0.141707219414880},
            point_t{0.307939838764121, 0.550352941820999},
            point_t{0.550352941820999, 0.141707219414880},
            point_t{0.550352941820999, 0.307939838764121},
            point_t{0.025003534762686, 0.246672560639903},
            point_t{0.025003534762686, 0.728323904597411},
            point_t{0.246672560639903, 0.025003534762686},
            point_t{0.246672560639903, 0.728323904597411},
            point_t{0.728323904597411, 0.025003534762686},
            point_t{0.728323904597411, 0.246672560639903},
            point_t{0.009540815400299, 0.066803251012200},
            point_t{0.009540815400299, 0.923655933587500},
            point_t{0.066803251012200, 0.009540815400299},
            point_t{0.066803251012200, 0.923655933587500},
            point_t{0.923655933587500, 0.009540815400299},
            point_t{0.923655933587500, 0.066803251012200},
        },
        .weights = {0.090817990382754,//
                    0.036725957756467,//
                    0.036725957756467,//
                    0.036725957756467,//
                    0.045321059435528,//
                    0.045321059435528,//
                    0.045321059435528,//
                    0.072757916845420,//
                    0.072757916845420,//
                    0.072757916845420,//
                    0.072757916845420,//
                    0.072757916845420,//
                    0.072757916845420,//
                    0.028327242531057,//
                    0.028327242531057,//
                    0.028327242531057,//
                    0.028327242531057,//
                    0.028327242531057,//
                    0.028327242531057,//
                    0.009421666963733,//
                    0.009421666963733,//
                    0.009421666963733,//
                    0.009421666963733,//
                    0.009421666963733,//
                    0.009421666963733}};
};

}// namespace wp::fields