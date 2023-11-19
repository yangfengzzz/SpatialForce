//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "io_mesh.h"
#include "../template_geometry.h"
#include <list>
#include <string>

namespace wp::fields {
/**
 * This class provides facilities to assess the mesh data file generated
 * by the mesh generator \p{gmsh}. For 3-dimensional only. Though we can
 * read in very flexible data format, we currently only use it to read in
 * pure tetrahedron mesh.
 */
class IOMesh1D : public IOMesh<1, 1> {
public:
    void read_mesh(float x0, float x1, uint32_t N);
};

}// namespace wp::fields