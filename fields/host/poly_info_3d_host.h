//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "poly_info_host.h"
#include "grid_host.h"
#include "../poly_info_3d.h"

namespace wp::fields {
template<int order>
class PolyInfo<Tetrahedron, order> {
public:
    static constexpr uint32_t n_unknown = (order + 3) * (order + 2) * (order + 1) / (3 * 2) - 1;

    poly_info_t<Tetrahedron, order> handle;

    explicit PolyInfo(GridPtr3D grid) : grid{std::move(grid)} {
        build_basis_func();
        sync_h2d();
    }

    ~PolyInfo();

private:
    void build_basis_func();
    void sync_h2d();

    GridPtr3D grid;
    std::vector<fixed_array_t<float, n_unknown>> poly_constants;
};

}// namespace wp::fields