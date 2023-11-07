//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/bvh.h"
#include "runtime/array.h"

namespace wp {
class BVH {
public:
    BVH(Stream& stream, const Array<vec3f>& lowers, const Array<vec3f>& uppers);

    ~BVH();

    void refit();

    static void refit(Stream& stream, bvh_t& bvh, const bounds3* b);

private:
    Stream& stream_;
    uint64_t id_{};
    bvh_t descriptor_{};
};
}  // namespace wp
