//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "mesh_host.h"

namespace wp::fields {
template class Mesh<1, 1>;
template class Mesh<2, 2>;
template class Mesh<3, 3>;
}
