//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "geometry_host.h"
#include "runtime/alloc.h"

namespace wp::fields {
uint32_t Geometry::n_index() const {
    return ind.size();
}

int32_t Geometry::index(uint32_t idx) const {
    return ind[idx];
}
uint32_t Geometry::n_vertex(uint32_t idx) const {
    if (idx == 0) {
        return vtx_index[0];
    } else {
        return vtx_index[idx] - vtx_index[idx - 1];
    }
}

uint32_t Geometry::vertex(uint32_t idx, uint32_t j) const {
    if (idx == 0) {
        return *(vtx.data() + j);
    } else {
        return *(vtx.data() + vtx_index[idx - 1] + j);
    }
}

uint32_t Geometry::n_boundary(uint32_t idx) const {
    if (idx == 0) {
        return bnd_index[0];
    } else {
        return bnd_index[idx] - bnd_index[idx - 1];
    }
}
uint32_t Geometry::boundary(uint32_t idx, uint32_t j) const {
    if (idx == 0) {
        return *(bnd.data() + j);
    } else {
        return *(bnd.data() + bnd_index[idx - 1] + j);
    }
}
int32_t Geometry::boundary_mark(uint32_t idx) const {
    return bm[idx];
}

void Geometry::sync_h2d() {
    handle.ind = alloc_array(ind);
    handle.vtx_index = alloc_array(vtx_index);
    handle.vtx = alloc_array(vtx);
    handle.bnd_index = alloc_array(bnd_index);
    handle.bnd = alloc_array(bnd);
    handle.bm = alloc_array(bm);
}

}// namespace wp::fields