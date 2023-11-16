//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "poly_info_host.h"
#include "grid_host.h"
#include "../poly_info_1d.h"
#include "core/mat.h"

namespace wp::fields {
template<int order>
class PolyInfo<1, order> {
public:
    static constexpr int n_unknown = order;
    using Mat = mat_t<n_unknown, n_unknown, float>;
    using Vec = mat_t<n_unknown, 1, float>;

    poly_info_t<1, order> handle;

private:
    GridPtr1D grid;
    std::vector<fixed_array_t<float, n_unknown>> poly_constants;
};

}// namespace wp::fields