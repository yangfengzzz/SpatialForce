//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/array.h"

namespace wp::fields {
/// the data to describe a geometry.
struct geometry_t {
    /// Index of the geometry.
    int32_t ind{};
    /// Index of vertices.
    array_t<uint32_t> vtx;
    /// Index of boundary geometries.
    array_t<uint32_t> bnd;

    /// An entry of the vertex index array.
    CUDA_CALLABLE uint32_t vertex(uint32_t i) const { return vtx[i]; }
    /// An entry of the boundary geometry index array.
    CUDA_CALLABLE uint32_t boundary(uint32_t i) const { return bnd[i]; }
};

struct geometry_bm_t : public geometry_t {
    using bmark_t = int;
    /// Boundary marker.
    bmark_t bm{};
};

template<uint32_t SIZE>
struct static_geometry_t {
    static constexpr uint32_t size = SIZE;
    /// Index of the geometry.
    int32_t ind{};
    /// Index of vertices.
    uint32_t vtx[size]{};
    /// Index of boundary geometries.
    uint32_t bnd[size]{};
};

}// namespace wp::fields
