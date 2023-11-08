//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/vec.h"
#include "fields/geometry.h"
#include "fields/quadrature_info.h"

namespace wp {
namespace fields {
/// The data structure of a mesh. The class \p Mesh administrate a set of points and
/// a set of geometries. The geometries are organized according its dimension and stored
/// in arrays. A lot of mechanism provided to retrieve information from the mesh.
template <uint32_t DIM, uint32_t DOW>
class mesh_t {
    /// Point array of the mesh.
    array_t<vec3> pnt;

    /// Geometries arrays of the mesh.
    /// The geometries in \p n dimension are in the \p n-th entry of the array,
    /// which is still an array.
    array_t<array_t<geometry_bm_t>> geo;
};

/// Template geometry is the geometry information of a template element. A template
/// geometry is in fact a one-element mesh. A template geometry have the information
/// about how to calculate its volume. Such a function is stored in a shared library.
/// The user should provide such a shared library and tell this class about the file
/// name of the shared library and the function name to calculate the volume.
template <uint32_t DIM>
class template_geometry_t : public mesh_t<DIM, DIM> {
    /// The quadrature information on the geometry.
    quadrature_info_admin_t<DIM> quad_info;
};

}  // namespace fields
}  // namespace wp