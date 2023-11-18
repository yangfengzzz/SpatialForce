//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "grid_system_data_host.h"

namespace wp::fields {
template<typename TYPE, uint32_t order, uint32_t dos>
GridSystemData<TYPE, order, dos>::GridSystemData(GridPtr<TYPE> grid)
    : _grid{grid} {}

template<typename TYPE, uint32_t order, uint32_t dos>
GridSystemData<TYPE, order, dos>::~GridSystemData() {
    for (int i = 0; i < dos; i++) {
        free_array(handle.scalar_data_list[i]);
    }
}

template<typename TYPE, uint32_t order, uint32_t dos>
void GridSystemData<TYPE, order, dos>::sync_h2d() {
    for (int i = 0; i < dos; i++) {
        handle.scalar_data_list[i] = alloc_array(_scalarDataList[i]);
    }
}

template<> class GridSystemData<Interval, 1, 1>;
template<> class GridSystemData<Triangle, 1, 1>;
template<> class GridSystemData<Tetrahedron, 1, 1>;

}// namespace wp::fields