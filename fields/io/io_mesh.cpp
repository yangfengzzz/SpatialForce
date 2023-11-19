//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "io_mesh.h"
#include <numeric>

namespace wp::fields {
bool is_same(const GeometryBM &g0, const GeometryBM &g1) {
    uint32_t i, j, k;
    j = g0.vtx.size();
    k = g1.vtx.size();
    if (j != k)
        return false;
    for (i = 0; i < k; i++) {
        for (j = 0; j < k; j++) {
            if (g0.vtx[i] == g1.vtx[j])
                break;
        }
        if (j == k)
            return false;
    }
    return true;
}

template<uint32_t DIM, uint32_t DOW>
Mesh<DIM, DOW> IOMesh<DIM, DOW>::create_mesh() {
    auto mesh = Mesh<DIM, DOW>{};
    // pnt
    mesh.pnt = point();

    for (int level = 0; level <= dim; level++) {
        const std::vector<GeometryBM> &geos = geometry(level);
        // vtx
        size_t total = 0;
        for (const auto &geo : geos) {
            total += geo.n_vertex();
        }
        std::vector<uint32_t> h_vtx_prefix_sum(geos.size());
        std::vector<uint32_t> vtx;
        vtx.reserve(total);
        for (size_t j = 0; j < geos.size(); j++) {
            h_vtx_prefix_sum[j] = geos[j].n_vertex();
            for (size_t k = 0; k < geos[j].n_vertex(); k++) {
                vtx.push_back(geos[j].vertex(k));
            }
        }
        mesh.geo[level].vtx_index = h_vtx_prefix_sum;
        std::inclusive_scan(mesh.geo[level].vtx_index.begin(), mesh.geo[level].vtx_index.end(),
                            mesh.geo[level].vtx_index.begin());
        mesh.geo[level].vtx = vtx;

        // bnd
        total = 0;
        for (const auto &geo : geos) {
            total += geo.n_boundary();
        }
        std::vector<uint32_t> h_bnd_prefix_sum(geos.size());
        std::vector<uint32_t> bnd;
        bnd.reserve(total);
        for (size_t j = 0; j < geos.size(); j++) {
            h_bnd_prefix_sum[j] = geos[j].n_boundary();
            for (size_t k = 0; k < geos[j].n_boundary(); k++) {
                bnd.push_back(geos[j].boundary(k));
            }
        }
        mesh.geo[level].bnd_index = h_bnd_prefix_sum;
        std::inclusive_scan(mesh.geo[level].bnd_index.begin(), mesh.geo[level].bnd_index.end(),
                            mesh.geo[level].bnd_index.begin());
        mesh.geo[level].bnd = bnd;

        // ind
        std::vector<int32_t> ind(geos.size());
        for (size_t j = 0; j < geos.size(); j++) {
            ind[j] = geos[j].index();
        }
        mesh.geo[level].ind = ind;

        // bm
        std::vector<int32_t> bm(geos.size());
        for (size_t j = 0; j < geos.size(); j++) {
            bm[j] = geos[j].boundaryMark();
        }
        mesh.geo[level].bm = bm;
    }
    return mesh;
}

template class IOMesh<1, 1>;
template class IOMesh<2, 1>;
template class IOMesh<3, 1>;

template class IOMesh<1, 2>;
template class IOMesh<2, 2>;
template class IOMesh<3, 2>;

template class IOMesh<1, 3>;
template class IOMesh<2, 3>;
template class IOMesh<3, 3>;
}// namespace wp::fields