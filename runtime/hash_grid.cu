//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/array.h"
#include "runtime/hash_grid.h"

namespace wp {
HashGrid::HashGrid(Device& device, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z) : device_{device} {}

}  // namespace wp