//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vector>
#include "../geometry.h"

namespace wp::fields {
class Geometry {
private:
    geometry_t handle;

    /// Index of the geometry.
    int32_t ind;
    /// Index of vertices.
    std::vector<uint32_t> vtx;
    /// Index of boundary geometries.
    std::vector<uint32_t> bnd;
};
}// namespace wp::fields