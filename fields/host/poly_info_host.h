//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <memory>

namespace wp::fields {

template<int dim, int order>
class PolyInfo {
    static_assert(dim < 1 || dim > 3, "Not implemented - N should be either 1, 2 or 3.");
};

template<int dim, int order>
using PolyInfoPtr = std::shared_ptr<PolyInfo<dim, order>>;

}// namespace wp::fields