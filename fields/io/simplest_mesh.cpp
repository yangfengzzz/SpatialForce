//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "simplest_mesh.h"

namespace wp::fields {
template class SimplestMesh<1, 1>;
template class SimplestMesh<2, 1>;
template class SimplestMesh<3, 1>;

template class SimplestMesh<1, 2>;
template class SimplestMesh<2, 2>;
template class SimplestMesh<3, 2>;

template class SimplestMesh<1, 3>;
template class SimplestMesh<2, 3>;
template class SimplestMesh<3, 3>;
}// namespace wp::fields