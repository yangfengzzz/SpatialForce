//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "../grid.h"
#include "../mesh.h"

namespace wp::fields {
template<uint32_t DIM>
class Grid {
public:
    static constexpr uint32_t dim = DIM;
    using point_t = vec_t<dim, float>;

    mesh_t<dim, dim> mesh_handle;
    grid_t<dim> grid_handle;
};
}// namespace wp::fields