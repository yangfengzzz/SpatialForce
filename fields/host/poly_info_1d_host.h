//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "poly_info_host.h"
#include "grid_host.h"
#include "../poly_info_1d.h"

namespace wp::fields {
template<int order>
class PolyInfo<1, order> {
public:
    static constexpr int n_unknown = order;

    poly_info_t<1, order> handle;

    explicit PolyInfo(GridPtr1D grid) : grid{std::move(grid)} {
        build_basis_func();
        sync_h2d();
    }

    ~PolyInfo();

private:
    void build_basis_func();
    void sync_h2d();

    GridPtr1D grid;
    std::vector<fixed_array_t<float, n_unknown>> poly_constants;
};

}// namespace wp::fields