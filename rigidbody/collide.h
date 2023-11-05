//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/array.h"
#include "core/spatial.h"
#include "model.h"

namespace wp {
inline CUDA_CALLABLE_DEVICE vec3f triangle_closest_point_barycentric(vec3f a, vec3f b, vec3f c, vec3f p) {
    auto ab = b - a;
    auto ac = c - a;
    auto ap = p - a;

    auto d1 = dot(ab, ap);
    auto d2 = dot(ac, ap);

    if (d1 <= 0.0 && d2 <= 0.0) {
        return {1.0, 0.0, 0.0};
    }

    auto bp = p - b;
    auto d3 = dot(ab, bp);
    auto d4 = dot(ac, bp);

    if (d3 >= 0.0 && d4 <= d3) {
        return {0.0, 1.0, 0.0};
    }
    auto vc = d1 * d4 - d3 * d2;
    auto v = d1 / (d1 - d3);
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        return {1.f - v, v, 0.0};
    }

    auto cp = p - c;
    auto d5 = dot(ab, cp);
    auto d6 = dot(ac, cp);

    if (d6 >= 0.0 && d5 <= d6) {
        return {0.0, 0.0, 1.0};
    }

    auto vb = d5 * d2 - d1 * d6;
    auto w = d2 / (d2 - d6);
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        return {1.f - w, 0.0, w};
    }

    auto va = d3 * d6 - d5 * d4;
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        return {0.0, w, 1.f - w};
    }

    auto denom = 1.f / (va + vb + vc);
    v = vb * denom;
    w = vc * denom;

    return {1.f - v - w, v, w};
}

void collide(Model& model, State& state, int edge_sdf_iter = 10);

}// namespace wp