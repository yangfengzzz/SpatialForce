//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "geometry.h"

namespace wp::fields {
CUDA_CALLABLE int32_t geometry_t::index(uint32_t idx) const {
    return ind[idx];
}

CUDA_CALLABLE uint32_t geometry_t::n_vertex(uint32_t idx) const {
    if (idx == 0) {
        return vtx_prefix_sum[0];
    } else {
        return vtx_prefix_sum[idx] - vtx_prefix_sum[idx - 1];
    }
}
CUDA_CALLABLE array_t<uint32_t> geometry_t::vertex(uint32_t idx) const {
    if (idx == 0) {
        return {vtx.data, vtx_prefix_sum[0]};
    } else {
        return {vtx.data + vtx_prefix_sum[idx - 1], vtx_prefix_sum[idx] - vtx_prefix_sum[idx - 1]};
    }
}
CUDA_CALLABLE uint32_t geometry_t::vertex(uint32_t idx, uint32_t j) const {
    if (idx == 0) {
        return *(vtx.data + j);
    } else {
        return *(vtx.data + vtx_prefix_sum[idx - 1] + j);
    }
}

CUDA_CALLABLE uint32_t geometry_t::n_boundary(uint32_t idx) const {
    if (idx == 0) {
        return bnd_prefix_sum[0];
    } else {
        return bnd_prefix_sum[idx] - bnd_prefix_sum[idx - 1];
    }
}
CUDA_CALLABLE array_t<uint32_t> geometry_t::boundary(uint32_t idx) const {
    if (idx == 0) {
        return {bnd.data, bnd_prefix_sum[0]};
    } else {
        return {bnd.data + bnd_prefix_sum[idx - 1], bnd_prefix_sum[idx] - bnd_prefix_sum[idx - 1]};
    }
}
CUDA_CALLABLE uint32_t geometry_t::boundary(uint32_t idx, uint32_t j) const {
    if (idx == 0) {
        return *(bnd.data + j);
    } else {
        return *(bnd.data + bnd_prefix_sum[idx - 1] + j);
    }
}
CUDA_CALLABLE int32_t geometry_t::boundary_mark(uint32_t idx) const {
    return bm[idx];
}
}// namespace wp::fields
