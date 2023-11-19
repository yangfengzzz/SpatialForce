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
/// This class provides facilities to assess the mesh data file generated
/// by the mesh generator \p{gmsh}. For 3-dimensional only. Though we can
/// read in very flexible data format, we currently only use it to read in
/// pure tetrahedron mesh.
class GmshMesh2D : public IOMesh<2, 2> {
public:
    using base_template_geometry_t = base_template_geometry_t<Triangle>;

    explicit GmshMesh2D(const std::string &);
    ~GmshMesh2D() override;

private:
    void parse_gmsh(const std::string &);
    void base_generate_mesh();
    void generate_mesh();

    std::list<GeometryBM> nodes;
    std::list<GeometryBM> lines;
    std::list<GeometryBM> surfaces;
};

}// namespace wp::fields