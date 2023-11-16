//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace wp::fields {

template<int dim, int order>
class poly_info_t {
    static_assert(dim < 1 || dim > 3, "Not implemented - N should be either 1, 2 or 3.");
};

}// namespace wp::fields