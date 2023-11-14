//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/array.h"
#include "core/fixed_array.h"

namespace wp::fields {
/// the data to describe a geometry.
struct geometry_t {
    /// Index of the geometry.
    int32_t ind{};
    /// Index of vertices.
    array_t<uint32_t> vtx;
    /// Index of boundary geometries.
    array_t<uint32_t> bnd;
    /// Boundary marker.
    int32_t bm{};

    /// An entry of the vertex index array.
    CUDA_CALLABLE uint32_t vertex(uint32_t i) const { return vtx[i]; }
    /// An entry of the boundary geometry index array.
    CUDA_CALLABLE uint32_t boundary(uint32_t i) const { return bnd[i]; }
};

template<uint32_t SIZE>
struct static_geometry_t {
    static constexpr uint32_t size = SIZE;
    /// Index of the geometry.
    int32_t ind{};
    /// Index of vertices.
    fixed_array_t<uint32_t, size> vtx{};
    /// Index of boundary geometries.
    fixed_array_t<uint32_t, size> bnd{};
};

}// namespace wp::fields
