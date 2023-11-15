//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "host/grid_host.h"
#include "runtime/alloc.h"
#include <fstream>
#include <vector>
#include <algorithm>

namespace wp::fields {
template class Grid<1>;
template class Grid<2>;
template class Grid<3>;

}// namespace wp::fields