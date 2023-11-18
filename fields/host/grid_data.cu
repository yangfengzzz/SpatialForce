//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "grid_data.h"

namespace wp::fields {
template<typename TYPE, uint32_t order>
GridData<TYPE, order>::GridData(uint32_t idx, GridPtr<TYPE> grid, ReconAuxiliaryPtr<TYPE, order> aux)
    : GridDataBase{idx}, grid{grid}, recon_auxiliary{aux}, handle{grid->grid_handle, aux->poly_info_handle()} {
}

template class GridData<Interval, 1>;
template class GridData<Triangle, 1>;
template class GridData<Tetrahedron, 1>;
}// namespace wp::fields