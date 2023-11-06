//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/hashgrid.h"

namespace wp {
class Device;

class HashGrid {
public:
    HashGrid(Device& device, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z);

private:
    Device& device_;
    hash_grid_t handle_;
};
}  // namespace wp