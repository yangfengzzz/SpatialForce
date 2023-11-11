//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "geometry.h"
#include "geometry_trait.h"
#include "mesh.h"

namespace wp::fields {
template<typename TEMPLATE_TYPE, uint32_t ACCURACY>
struct SurfaceIntegrator {
    static constexpr uint32_t tdim = TEMPLATE_TYPE::tdim;
    using CoordTransform = TEMPLATE_TYPE;
    using TemplateGeomtry = template_geometry_t<typename TEMPLATE_TYPE::associate_t, ACCURACY>;
    mesh_t<tdim, tdim> mesh;

    template<typename FUNCTOR>
    CUDA_CALLABLE FUNCTOR::RETURN_TYPE operator()(uint32_t index, FUNCTOR functor) {
        constexpr uint32_t arr_len = CoordTransform::arr_len;
        vec_t<tdim, float> arr[arr_len];
        for (uint32_t i = 0; i < arr_len; i++) {
            // global mem to local register
            arr[i] = mesh.pnt[mesh.geo[tdim - 1][index].vertex(i)];
        }

        constexpr uint32_t n_quad_size = TemplateGeomtry::n_quad_size;
        constexpr auto quadrature_info = TemplateGeomtry::quadrature_info();
        constexpr auto template_points = TemplateGeomtry::points();
        typename FUNCTOR::RETURN_TYPE result{};
        for (uint32_t i = 0; i < n_quad_size; i++) {
            auto quad_pt = quadrature_info.pnts[i];
            auto pt = CoordTransform::local_to_global(quad_pt, template_points.pnts, arr);
            auto jxw = CoordTransform::local_to_global_jacobian(quad_pt, template_points.pnts, arr);
            jxw *= quadrature_info.weights[i] * CoordTransform::volume(template_points.pnts);
            result += jxw * functor(pt);
        }

        return result;
    }
};

template<uint32_t ACCURACY>
struct SurfaceIntegrator<Interval, ACCURACY> {
    array_t<vec_t<1, float>> pnt;

    template<typename FUNCTOR>
    CUDA_CALLABLE FUNCTOR::RETURN_TYPE operator()(uint32_t i, FUNCTOR functor) {
        return functor(pnt[i]);
    }
};

}// namespace wp::fields