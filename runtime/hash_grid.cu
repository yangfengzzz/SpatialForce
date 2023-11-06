//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/array.h"
#include "runtime/cuda_util.h"
#include "runtime/hash_grid.h"

namespace wp {
__global__ void compute_cell_indices(hash_grid_t grid, const wp::vec3* points, int num_points) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_points) {
        grid.point_cells[tid] = hash_grid_index(grid, points[tid]);
        grid.point_ids[tid] = tid;
    }
}

__global__ void compute_cell_offsets(int* cell_starts, int* cell_ends, const int* point_cells, int num_points) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // compute cell start / end
    if (tid < num_points) {
        // scan the particle-cell array to find the start and end
        const int c = point_cells[tid];

        if (tid == 0)
            cell_starts[c] = 0;
        else {
            const int p = point_cells[tid - 1];

            if (c != p) {
                cell_starts[c] = tid;
                cell_ends[p] = tid;
            }
        }

        if (tid == num_points - 1) {
            cell_ends[c] = tid + 1;
        }
    }
}

HashGrid::HashGrid(Stream& stream, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z) {
    hash_grid_t grid;
    memset(&grid, 0, sizeof(HashGrid));

    grid.dim_x = dim_x;
    grid.dim_y = dim_y;
    grid.dim_z = dim_z;

    const int num_cells = dim_x * dim_y * dim_z;
    grid.cell_starts = (int*)Device::alloc(num_cells * sizeof(int));
    grid.cell_ends = (int*)Device::alloc(num_cells * sizeof(int));

    // upload to device
    auto* grid_device = (HashGrid*)(Device::alloc(sizeof(HashGrid)));
    stream.memcpy_h2d(grid_device, &grid, sizeof(HashGrid));

    grid_id_ = (uint64_t)(grid_device);
}

void HashGrid::rebuild(Stream& stream, const wp::vec3* points, int num_points) {
    wp_launch_device(nullptr, wp::compute_cell_indices, num_points, (handle_, points, num_points));

    RadixSort::sort_pairs(stream, handle_.point_cells, handle_.point_ids, num_points);

    const int num_cells = handle_.dim_x * handle_.dim_y * handle_.dim_z;

    stream.memset(handle_.cell_starts, 0, sizeof(int) * num_cells);
    stream.memset(handle_.cell_ends, 0, sizeof(int) * num_cells);

    wp_launch_device(nullptr, wp::compute_cell_offsets, num_points,
                     (handle_.cell_starts, handle_.cell_ends, handle_.point_cells, num_points));
}
}  // namespace wp