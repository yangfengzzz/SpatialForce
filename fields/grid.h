//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/vec.h"
#include "core/fixed_array.h"
#include "fields/geometry.h"
#include "geometry_trait.h"
#include "template_geometry.h"
#include "surface_integrator.h"
#include "volume_integrator.h"

namespace wp::fields {
template<typename TYPE>
struct grid_base_t {
    static constexpr uint32_t dim = TYPE::dim;
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

template<typename TYPE>
struct grid_t : public grid_base_t<TYPE> {
};

template<>
struct grid_t<Interval> : public grid_base_t<Interval> {
    VolumeIntegrator<Interval, 1> volume_integrator;
    SurfaceIntegrator<Interval, 1> surface_integrator;
};

template<>
struct grid_t<Triangle> : public grid_base_t<Triangle> {
    VolumeIntegrator<Triangle, 1> volume_integrator;
    SurfaceIntegrator<IntervalTo2D, 1> surface_integrator;
};

template<>
struct grid_t<Tetrahedron> : public grid_base_t<Tetrahedron> {
    VolumeIntegrator<Tetrahedron, 1> volume_integrator;
    SurfaceIntegrator<TriangleTo3D, 1> surface_integrator;
};

}// namespace wp::fields