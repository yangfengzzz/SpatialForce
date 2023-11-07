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

    ~HashGrid();

    uint64_t id() const { return grid_id_; }

    void reserve(int num_points);

    void update(float cell_width, const Array<wp::vec3>& positions, int num_points);

private:
    Stream& stream_;
    RadixSort sort_;
    hash_grid_t handle_;
    uint64_t grid_id_;

    void rebuild(const Array<wp::vec3>& points, int num_points);
};
}  // namespace wp