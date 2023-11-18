//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "recon_auxiliary_host.h"
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace wp::fields {
namespace {
template<typename TYPE, uint32_t Order>
struct UpdateLSMatrixFunctor {
    CUDA_CALLABLE explicit UpdateLSMatrixFunctor(grid_t<TYPE> grid, poly_info_t<TYPE, Order> poly, recon_auxiliary_t<TYPE, Order> aux)
        : functor(grid, poly), aux_view(aux) {}

    template<typename Index>
    inline CUDA_CALLABLE void operator()(Index ele_idx) {
        functor(ele_idx, aux_view.get_patch(ele_idx), aux_view.get_poly_avgs(ele_idx), aux_view.get_g_inv(ele_idx));

        aux_view.G_inv[ele_idx] = inverse(aux_view.G_inv[ele_idx]);
        // todo larger matrix inverse for high order reconstruction
    }

private:
    recon_auxiliary_t<TYPE, Order> aux_view;
    typename poly_info_t<TYPE, Order>::UpdateLSMatrixFunctor functor;
};
}// namespace

template<typename TYPE, uint32_t Order>
void ReconAuxiliary<TYPE, Order>::build_ls_matrix() {
    thrust::for_each(thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(0) + grid->n_geometry(TYPE::dim),
                     UpdateLSMatrixFunctor<TYPE, Order>(grid->grid_handle, polyInfo.handle, handle));
}

template<typename TYPE, uint32_t Order>
void ReconAuxiliary<TYPE, Order>::sync_h2d() {
    handle.patch_prefix_sum = alloc_array(patch_prefix_sum);
    handle.patch = alloc_array(patch);
    handle.patch_polys = alloc_array(patch_polys);
    handle.G_inv = alloc_array(G_inv);
}

template<typename TYPE, uint32_t Order>
ReconAuxiliary<TYPE, Order>::ReconAuxiliary(GridPtr<TYPE> grid)
    : grid{grid}, polyInfo{grid} {
    build_ls_matrix();
    sync_h2d();
}

template<typename TYPE, uint32_t Order>
ReconAuxiliary<TYPE, Order>::~ReconAuxiliary() {
    free_array(handle.patch_prefix_sum);
    free_array(handle.patch);
    free_array(handle.patch_polys);
    free_array(handle.G_inv);
}

template class ReconAuxiliary<Interval, 1>;
template class ReconAuxiliary<Interval, 2>;
template class ReconAuxiliary<Interval, 3>;

template class ReconAuxiliary<Triangle, 1>;

template class ReconAuxiliary<Tetrahedron, 1>;
}// namespace wp::fields