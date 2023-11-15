//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "../grid.h"
#include "../mesh.h"
#include <string>
#include "mesh_host.h"
#include "runtime/alloc.h"

namespace wp::fields {
template<uint32_t DIM>
class Grid {
public:
    static constexpr uint32_t dim = DIM;
    using point_t = vec_t<dim, float>;

    mesh_t<dim, dim> mesh_handle;
    grid_t<dim> grid_handle;

    void sync_h2d() {
        grid_handle.bary_center = alloc_from_vector(bary_center);
        grid_handle.volume = alloc_from_vector(volume);
        grid_handle.size = alloc_from_vector(size);
        grid_handle.neighbour = alloc_from_vector(neighbour);
        grid_handle.period_bry = alloc_from_vector(period_bry);
        grid_handle.boundary_center = alloc_from_vector(boundary_center);
        grid_handle.bry_size = alloc_from_vector(bry_size);
        grid_handle.boundary_mark = alloc_from_vector(boundary_mark);

        mesh.sync_h2d();
        mesh_handle = mesh.handle;
    }

private:
    Mesh<dim, dim> mesh;
    std::vector<point_t> bary_center;
    std::vector<float> volume;
    std::vector<float> size;

    std::vector<fixed_array_t<int32_t, 2>> neighbour;
    std::vector<fixed_array_t<int32_t, 2>> period_bry;
    std::vector<point_t> boundary_center;
    std::vector<float> bry_size;
    std::vector<int32_t> boundary_mark;
};
}// namespace wp::fields