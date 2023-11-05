//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "vec.h"

namespace wp {
struct bounds3 {
    CUDA_CALLABLE inline bounds3() : lower(FLT_MAX), upper(-FLT_MAX) {}

    CUDA_CALLABLE inline bounds3(const vec3& lower, const vec3& upper) : lower(lower), upper(upper) {}

    CUDA_CALLABLE inline vec3 center() const { return 0.5f * (lower + upper); }
    CUDA_CALLABLE inline vec3 edges() const { return upper - lower; }

    CUDA_CALLABLE inline void expand(float r) {
        lower -= vec3(r);
        upper += vec3(r);
    }

    CUDA_CALLABLE inline void expand(const vec3& r) {
        lower -= r;
        upper += r;
    }

    CUDA_CALLABLE inline bool empty() const {
        return lower[0] >= upper[0] || lower[1] >= upper[1] || lower[2] >= upper[2];
    }

    CUDA_CALLABLE inline bool overlaps(const vec3& p) const {
        if (p[0] < lower[0] || p[1] < lower[1] || p[2] < lower[2] || p[0] > upper[0] || p[1] > upper[1] ||
            p[2] > upper[2]) {
            return false;
        } else {
            return true;
        }
    }

    CUDA_CALLABLE inline bool overlaps(const bounds3& b) const {
        if (lower[0] > b.upper[0] || lower[1] > b.upper[1] || lower[2] > b.upper[2] || upper[0] < b.lower[0] ||
            upper[1] < b.lower[1] || upper[2] < b.lower[2]) {
            return false;
        } else {
            return true;
        }
    }

    CUDA_CALLABLE inline void add_point(const vec3& p) {
        lower = min(lower, p);
        upper = max(upper, p);
    }

    CUDA_CALLABLE inline float area() const {
        vec3 e = upper - lower;
        return 2.0f * (e[0] * e[1] + e[0] * e[2] + e[1] * e[2]);
    }

    vec3 lower;
    vec3 upper;
};

CUDA_CALLABLE inline bounds3 bounds_union(const bounds3& a, const vec3& b) {
    return bounds3(min(a.lower, b), max(a.upper, b));
}

CUDA_CALLABLE inline bounds3 bounds_union(const bounds3& a, const bounds3& b) {
    return bounds3(min(a.lower, b.lower), max(a.upper, b.upper));
}

CUDA_CALLABLE inline bounds3 bounds_intersection(const bounds3& a, const bounds3& b) {
    return bounds3(max(a.lower, b.lower), min(a.upper, b.upper));
}

}  // namespace wp