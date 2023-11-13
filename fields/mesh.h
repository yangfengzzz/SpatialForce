//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/vec.h"
#include "core/fixed_array.h"
#include "fields/geometry.h"

namespace wp::fields {
/// The data structure of a mesh. The class \p Mesh administrate a set of points and
/// a set of geometries. The geometries are organized according its dimension and stored
/// in arrays. A lot of mechanism provided to retrieve information from the mesh.
template<uint32_t DIM, uint32_t DOW>
struct mesh_t {
    static constexpr uint32_t dim = DIM;
    static constexpr uint32_t dow = DOW;

    /// Point array of the mesh.
    array_t<vec_t<dow, float>> pnt;

    /// Geometries arrays of the mesh.
    /// The geometries in \p n dimension are in the \p n-th entry of the array,
    /// which is still an array.
    fixed_array_t<array_t<geometry_bm_t>, dim + 1> geo;

    /// Geometries array in certain dimension.
    [[nodiscard]] CUDA_CALLABLE const array_t<geometry_bm_t> &geometry(int d) const {
        return geo[d];
    }

    /// Geometries array in certain dimension.
    CUDA_CALLABLE array_t<geometry_bm_t> &geometry(int d) {
        return geo[d];
    }

    /// Certain geometry in certain dimension.
    [[nodiscard]] CUDA_CALLABLE const geometry_bm_t &geometry(int d, int index) const {
        return geo[d](index);
    }

    /// Certain geometry in certain dimension.
    CUDA_CALLABLE geometry_bm_t &geometry(int d, int index) {
        return geo[d](index);
    }

    /// Boundary marker of certain geometry in certain dimension.
    [[nodiscard]] CUDA_CALLABLE geometry_bm_t::bmark_t boundaryMark(int d, int index) const {
        return geo[d](index).bm;
    }

    /// Boundary marker of certain geometry in certain dimension.
    CUDA_CALLABLE geometry_bm_t::bmark_t &boundaryMark(int d, int index) {
        return geo[d](index).bm;
    }
};

}// namespace wp::fields
