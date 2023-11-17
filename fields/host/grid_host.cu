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
template class Grid<Interval>;
template class Grid<Triangle>;
template class Grid<Tetrahedron>;

}// namespace wp::fields