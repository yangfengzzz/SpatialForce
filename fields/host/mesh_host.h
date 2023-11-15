//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "../mesh.h"
#include "geometry_host.h"
#include <vector>

namespace wp::fields {
template<uint32_t DIM, uint32_t DOW>
class Mesh {
public:
    static constexpr uint32_t dim = DIM;
    static constexpr uint32_t dow = DOW;
    using point_t = vec_t<dow, float>;

    /// Number of points in the mesh.
    [[nodiscard]] uint32_t n_point() const {
        return pnt.size();
    }

    /// Number of geometries in certain dimension.
    [[nodiscard]] uint32_t n_geometry(int n) const {
        return geo[n].n_index();
    }

    /// Point array.
    const std::vector<point_t> &point() const {
        return pnt;
    }

    /// Point array.
    std::vector<point_t> &point() {
        return pnt;
    }

    /// A certain point.
    const point_t &point(int i) const {
        return pnt[i];
    }

    /// A certain point.
    point_t &point(int i) {
        return pnt[i];
    }

    /// Geometries array in certain dimension.
    [[nodiscard]] const Geometry &geometry(int n) const {
        return geo[n];
    }

    /// Geometries array in certain dimension.
    Geometry &geometry(int n) {
        return geo[n];
    }

    /// Boundary marker of certain geometry in certain dimension.
    [[nodiscard]] int32_t boundary_mark(int n, int j) const {
        return geo[n].boundary_mark(j);
    }

private:
    mesh_t<dim, dow> handle;

    /// Point array of the mesh.
    std::vector<point_t> pnt;
    /// Geometries arrays of the mesh.
    /// The geometries in \p n dimension are in the \p n-th entry of the array,
    /// which is still an array.
    Geometry geo[dim + 1];
};

}// namespace wp::fields