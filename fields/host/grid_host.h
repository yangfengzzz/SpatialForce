//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "../grid.h"
#include "../mesh.h"
#include <string>

namespace wp::fields {
template<uint32_t DIM>
class Grid {
public:
    static constexpr uint32_t dim = DIM;
    using point_t = vec_t<dim, float>;

    explicit Grid(const std::string &filename, bool periodic = false);

    mesh_t<dim, dim> mesh_handle;
    grid_t<dim> grid_handle;

private:
    mesh_t<dim, dim> read_data(const std::string &filename) {
        return mesh_t<dim, dim>{};
    }

    mesh_t<dim, dim> read_data_1d(const std::string &filename);
};
}// namespace wp::fields