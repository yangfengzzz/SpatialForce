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
    array_t<int32_t> ind{};
    /// Index of vertices.
    array_t<int32_t> vtx_prefix_sum;
    array_t<uint32_t> vtx;
    /// Index of boundary geometries.
    array_t<int32_t> bnd_prefix_sum;
    array_t<uint32_t> bnd;
    /// Boundary marker.
    array_t<int32_t> bm{};

    CUDA_CALLABLE int32_t index(uint32_t) const;
    /// Number of vertices.
    CUDA_CALLABLE uint32_t n_vertex(uint32_t) const;
    /// The vertex index array.
    CUDA_CALLABLE array_t<uint32_t> vertex(uint32_t) const;
    /// An entry of the vertex index array.
    CUDA_CALLABLE uint32_t vertex(uint32_t, uint32_t) const;

    /// Number of boundary geometries.
    CUDA_CALLABLE uint32_t n_boundary(uint32_t) const;
    /// The boundary geometry index array.
    CUDA_CALLABLE array_t<uint32_t> boundary(uint32_t) const;
    /// An entry of the boundary geometry index array.
    CUDA_CALLABLE uint32_t boundary(uint32_t, uint32_t) const;
    /// Access to the boundary marker.
    CUDA_CALLABLE int32_t boundary_mark(uint32_t) const;
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
