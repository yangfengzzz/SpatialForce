//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "host/grid_host.h"
#include "runtime/alloc.h"
#include <fstream>
#include <vector>
#include <algorithm>

namespace wp::fields {
template<>
mesh_t<1, 1> Grid<1>::read_data_1d(const std::string &filename) {
    int i, j;
    std::ifstream is(filename.c_str());
    is >> i;
    std::vector<float> buffer(i);
    for (j = 0; j < i; j++)
        is >> buffer[j];
    is.close();
    std::sort(buffer.begin(), buffer.end());

    mesh_t<1, 1> mesh;
    auto pnt_ptr = (vec_t<1, float> *)alloc_host(sizeof(vec_t<1, float>) * i);
    mesh.pnt = array_t<vec_t<1, float>>{pnt_ptr, 1};
    for (j = 0; j < i; j++)
        mesh.pnt[0] = buffer[j];

    auto geometry_ptr1 = (geometry_t *)alloc_host(sizeof(geometry_t));

    return mesh;
}

template<>
Grid<1>::Grid(const std::string &filename, bool periodic) {
    auto mesh_host = read_data_1d(filename);
}

}// namespace wp::fields