//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "simplest_mesh.h"
#include <list>
#include <string>

namespace wp::fields {
/**
 * This class provides facilities to assess the mesh data file generated
 * by the mesh generator \p{gmsh}. For 3-dimensional only. Though we can
 * read in very flexible data format, we currently only use it to read in
 * pure tetrahedron mesh.
 */
class GmshMesh2D : public SimplestMesh<2, 2> {
public:
    using GeometryType = int;
    static constexpr GeometryType POINT = 15;
    static constexpr GeometryType LINE = 1;
    static constexpr GeometryType TRIANGLE = 2;
    enum {
        N_POINT_NODE = 1,
        N_LINE_NODE = 2,
        N_TRIANGLE_NODE = 3,
        N_QUADRANGLE_NODE = 4,
        N_TETRAHEDRON_NODE = 4,
        N_HEXAHEDRON_NODE = 8,
        N_PRISM_NODE = 6,
        N_PYRAMID_NODE = 5,
    };

    struct GeometryBM {
        /// Index of the geometry.
        int ind;
        /// Index of vertices.
        std::vector<uint32_t> vtx;
        /// Index of boundary geometries.
        std::vector<uint32_t> bnd;
        /// Boundary marker.
        int bm;
    };
    std::list<GeometryBM> nodes;
    std::list<GeometryBM> lines;
    std::list<GeometryBM> surfaces;

public:
    GmshMesh2D();
    ~GmshMesh2D() override;

public:
    void read_data(const std::string &);
};

}// namespace wp::fields