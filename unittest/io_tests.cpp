//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <gtest/gtest.h>
#include "fields/io/gmsh2d_io.h"

using namespace wp::fields;

TEST(io, mes2D) {
    GmshMesh2D loader;
    loader.read_data("grids/2d/diagsquare.msh");
    auto mesh = loader.create_mesh();
    mesh.sync_h2d();
}