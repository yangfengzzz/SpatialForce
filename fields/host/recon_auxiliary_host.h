//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "../recon_auxiliary.h"
#include "poly_info_1d_host.h"
#include "poly_info_2d_host.h"
#include "poly_info_3d_host.h"
#include <memory>
#include <vector>

namespace wp::fields {

template<typename TYPE, uint32_t Order>
class ReconAuxiliary {
public:
    recon_auxiliary_t<TYPE, Order> handle;

    ReconAuxiliary(GridPtr<TYPE> grid);

    ~ReconAuxiliary();

private:
    void build_ls_matrix();
    void sync_h2d();

    GridPtr<TYPE> grid;
    PolyInfo<TYPE, Order> polyInfo;
    struct ElementHelper {
        std::vector<int32_t> patch;
    };
    std::vector<ElementHelper> element_helper;

    std::vector<uint32_t> patch_prefix_sum;
    std::vector<int32_t> patch;
    std::vector<fixed_array_t<float, poly_info_t<TYPE, Order>::n_unknown>> patch_polys;
    std::vector<typename poly_info_t<TYPE, Order>::Mat> G_inv;
};

template<typename TYPE, uint32_t Order>
using ReconAuxiliaryPtr = std::shared_ptr<ReconAuxiliary<TYPE, Order>>;

}// namespace wp::fields