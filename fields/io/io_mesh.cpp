//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "io_mesh.h"

namespace wp::fields {
bool is_same(const GeometryBM &g0, const GeometryBM &g1) {
    uint32_t i, j, k;
    j = g0.vtx.size();
    k = g1.vtx.size();
    if (j != k)
        return false;
    for (i = 0; i < k; i++) {
        for (j = 0; j < k; j++) {
            if (g0.vtx[i] == g1.vtx[j])
                break;
        }
        if (j == k)
            return false;
    }
    return true;
}

template class IOMesh<1, 1>;
template class IOMesh<2, 1>;
template class IOMesh<3, 1>;

template class IOMesh<1, 2>;
template class IOMesh<2, 2>;
template class IOMesh<3, 2>;

template class IOMesh<1, 3>;
template class IOMesh<2, 3>;
template class IOMesh<3, 3>;
}// namespace wp::fields