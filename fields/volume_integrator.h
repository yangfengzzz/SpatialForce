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
struct VolumeIntegrator {
    static constexpr uint32_t dim = TEMPLATE_TYPE::dim;
    using CoordTransform = TEMPLATE_TYPE;
    using TemplateGeomtry = template_geometry_t<TEMPLATE_TYPE, ACCURACY>;
    mesh_t<dim, dim> mesh;

    template<typename FUNCTOR>
    CUDA_CALLABLE FUNCTOR::RETURN_TYPE operator()(uint32_t index, FUNCTOR functor) {
        constexpr uint32_t arr_len = CoordTransform::arr_len;
        vec_t<dim, float> arr[arr_len];
        for (uint32_t i = 0; i < arr_len; i++) {
            // global mem to local register
            arr[i] = mesh.pnt[mesh.geo[dim].vertex(index, i)];
        }

        constexpr auto quadrature_info = TemplateGeomtry::quadrature_info();
        constexpr uint32_t n_quad_size = quadrature_info.size;
        constexpr auto template_points = TemplateGeomtry::points();
        typename FUNCTOR::RETURN_TYPE result{};
        for (uint32_t i = 0; i < n_quad_size; i++) {
            auto quad_pt = quadrature_info.pnts[i];
            auto pt = CoordTransform::local_to_global(quad_pt, template_points.data, arr);
            auto jxw = CoordTransform::local_to_global_jacobian(quad_pt, template_points.data, arr);
            jxw *= quadrature_info.weights[i] * CoordTransform::volume(template_points.data);
            result += jxw * functor(pt);
        }

        return result;
    }
};

}// namespace wp::fields