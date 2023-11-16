//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "poly_info.h"

namespace wp::fields {

template<int order>
struct poly_info_t<1, order> {
    static constexpr int n_unknown = order;
    array_t<fixed_array_t<float, n_unknown>> poly_constants;
};

}// namespace wp::fields