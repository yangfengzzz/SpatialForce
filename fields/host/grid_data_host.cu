//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "grid_data_host.h"

namespace wp::fields {
template<typename TYPE>
GridDataSimple<TYPE>::~GridDataSimple() {
    free_array(handle.data);
}

template<typename TYPE>
void GridDataSimple<TYPE>::sync_h2d() {
    handle.data = alloc_array(data);
}

template class GridDataSimple<Interval>;
template class GridDataSimple<Triangle>;
template class GridDataSimple<Tetrahedron>;

template<typename TYPE, uint32_t order>
GridData<TYPE, order>::GridData(uint32_t idx, GridPtr<TYPE> grid, ReconAuxiliaryPtr<TYPE, order> aux)
    : GridDataBase{idx}, grid{grid}, recon_auxiliary{aux} {
}

template<typename TYPE, uint32_t order>
GridData<TYPE, order>::~GridData() {
    free_array(handle.data);
    free_array(handle.slope);
}

template<typename TYPE, uint32_t order>
void GridData<TYPE, order>::sync_h2d() {
    handle.data = alloc_array(data);
    handle.slope = alloc_array(slope);
}

template class GridData<Interval, 1>;
template class GridData<Triangle, 1>;
template class GridData<Tetrahedron, 1>;
}// namespace wp::fields