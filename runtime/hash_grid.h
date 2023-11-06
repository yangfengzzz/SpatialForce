//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/hashgrid.h"
#include "runtime/radix_sort.h"

namespace wp {
class Device;

class HashGrid {
public:
    HashGrid(Stream& stream, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z);

private:
    hash_grid_t handle_;
    uint64_t grid_id_;

    void rebuild(Stream& stream, const wp::vec3* points, int num_points);
};
}  // namespace wp