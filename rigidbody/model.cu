//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "model.h"

namespace wp {
std::tuple<float, vec3f, mat33f> compute_shape_mass(GeometryType type, float scale, float density, bool is_solid, float thickness) {
    if (density == 0 || type == GeometryType::GEO_PLANE) {
        return std::make_tuple(0, vec3f(), mat33f());
    }

    if (type == GeometryType::GEO_SPHERE) {

    } else if (type == GeometryType::GEO_BOX) {

    } else if (type == GeometryType::GEO_CAPSULE) {
    } else if (type == GeometryType::GEO_CYLINDER) {

    } else if (type == GeometryType::GEO_CONE) {
    }

    return std::make_tuple(0, vec3f(), mat33f());
}

std::tuple<float, vec3f, mat33f> compute_shape_mass(GeometryType type, SDF &src, float scale, float density, bool is_solid, float thickness) {
    if (src.has_inertia && src.mass > 0 && src.is_solid == is_solid) {
        auto m = src.mass;
        auto c = src.com;
        auto I = src.I;
    }

    return std::make_tuple(0, vec3f(), mat33f());
}
}