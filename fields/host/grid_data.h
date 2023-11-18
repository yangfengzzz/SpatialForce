//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "grid_host.h"
#include "recon_auxiliary_host.h"
#include "../grid_data.h"

namespace wp::fields {
class GridDataBase {
public:
    explicit GridDataBase(uint32_t idx) : idx{idx} {}

protected:
    const uint32_t idx;
};

template<typename TYPE, uint32_t order>
class GridData : GridDataBase {
public:
    using point_t = typename Grid<TYPE>::point_t;

    GridData(uint32_t idx, GridPtr<TYPE> grid, ReconAuxiliaryPtr<TYPE, order> aux);

    grid_data_t<TYPE, order> handle;

private:
    const ReconAuxiliaryPtr<TYPE, order> recon_auxiliary;
    std::vector<typename poly_info_t<TYPE, order>::Vec> slope;
    const GridPtr<TYPE> grid;
};
}// namespace wp::fields