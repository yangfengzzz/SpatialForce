//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/vec.h"
#include "fields/geometry.h"

namespace wp::fields {
/// The data structure of a mesh. The class \p Mesh administrate a set of points and
/// a set of geometries. The geometries are organized according its dimension and stored
/// in arrays. A lot of mechanism provided to retrieve information from the mesh.
template<uint32_t dim, uint32_t dow>
class mesh_t {
    /// Point array of the mesh.
    array_t<vec_t<dow, float>> pnt;

    /// Geometries arrays of the mesh.
    /// The geometries in \p n dimension are in the \p n-th entry of the array,
    /// which is still an array.
    array_t<geometry_bm_t> geo[dim + 1];
};

}// namespace wp::fields
