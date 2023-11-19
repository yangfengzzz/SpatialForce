//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "io_mesh_1d.h"

namespace wp::fields {
void IOMesh1D::read_mesh(float x0, float x1, uint32_t N) {
    uint32_t i = N + 1;
    float dx = (x1 - x0) / float(N);

    point().resize(i);
    for (int j = 0; j < i; j++)
        point(j)[0] = x0 + float(i) * dx;

    geometry(0).resize(i);
    boundaryMark(0, 0) = 1;
    boundaryMark(0, i - 1) = 1;
    for (int j = 0; j < i; j++) {
        geometry(0, j).index() = j;
        geometry(0, j).vertex().resize(1, j);
        geometry(0, j).boundary().resize(1, j);
    }
    geometry(1).resize(i - 1);
    for (int j = 0; j < i - 1; j++) {
        geometry(1, j).index() = j;
        geometry(1, j).vertex().resize(2);
        geometry(1, j).vertex(0) = j;
        geometry(1, j).vertex(1) = j + 1;
        geometry(1, j).boundary().resize(2);
        geometry(1, j).boundary(0) = j;
        geometry(1, j).boundary(1) = j + 1;
    }
}

}// namespace wp::fields