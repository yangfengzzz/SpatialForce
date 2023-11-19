//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vector>
#include "core/vec.h"
#include "host/mesh_host.h"

namespace wp::fields {
struct GeometryBM {
    /// Index of the geometry.
    int ind{};
    /// Index of vertices.
    std::vector<uint32_t> vtx;
    /// Index of boundary geometries.
    std::vector<uint32_t> bnd;
    /// Boundary marker.
    int bm{};

    [[nodiscard]] inline int index() const {
        return ind;
    }
    inline int &index() {
        return ind;
    }
    /// Number of vertices.
    [[nodiscard]] inline uint32_t n_vertex() const {
        return vtx.size();
    }
    /// Number of boundary geometries.
    [[nodiscard]] inline uint32_t n_boundary() const {
        return bnd.size();
    }
    /// The vertex index array.
    [[nodiscard]] inline const std::vector<uint32_t> &vertex() const {
        return vtx;
    }
    /// The vertex index array.
    inline std::vector<uint32_t> &vertex() {
        return vtx;
    }
    /// The boundary geometry index array.
    [[nodiscard]] inline const std::vector<uint32_t> &boundary() const {
        return bnd;
    }
    /// The boundary geometry index array.
    inline std::vector<uint32_t> &boundary() {
        return bnd;
    }
    /// An entry of the vertex index array.
    [[nodiscard]] inline uint32_t vertex(uint32_t idx) const {
        return vtx[idx];
    }
    /// An entry of the vertex index array.
    inline uint32_t &vertex(uint32_t idx) {
        return vtx[idx];
    }
    /// An entry of the boundary geometry index array.
    [[nodiscard]] inline uint32_t boundary(uint32_t idx) const {
        return bnd[idx];
    }
    /// An entry of the boundary geometry index array.
    inline uint32_t &boundary(uint32_t idx) {
        return bnd[idx];
    }
    /// Access to the boundary marker.
    [[nodiscard]] inline int boundaryMark() const {
        return bm;
    }
    /// Access to the boundary marker.
    inline int &boundaryMark() {
        return bm;
    }

    /// Judge if two geometries are the same.
    friend bool is_same(const GeometryBM &, const GeometryBM &);
};

bool is_same(const GeometryBM &g0, const GeometryBM &g1);

/**
 * IO mesh is a kind of mesh with only the points coordinate and the element
 * geometry information. Some grid generation program provide such kind of data format.
 * This class provides facilities to generate a mesh with internal data format from a
 * simplest mesh. This class will be helpful when the data provided by grid generation
 * program. Generally, a grid generation program will be sure to provide this class
 * required. Warning: to generate a mesh with internal data format will be really
 * time-consuming.
 */
template<uint32_t DIM, uint32_t DOW = DIM>
class IOMesh {
public:
    using point_t = vec_t<DOW, float>;
    static constexpr uint32_t dim = DIM;
    static constexpr uint32_t dow = DOW;

    using GeometryType = int;
    static constexpr GeometryType POINT = 15;
    static constexpr GeometryType LINE = 1;
    static constexpr GeometryType TRIANGLE = 2;
    static constexpr GeometryType QUADRANGLE = 3;
    static constexpr GeometryType TETRAHEDRON = 4;
    static constexpr GeometryType HEXAHEDRON = 5;

    enum {
        N_POINT_NODE = 1,
        N_LINE_NODE = 2,
        N_TRIANGLE_NODE = 3,
        N_QUADRANGLE_NODE = 4,
        N_TETRAHEDRON_NODE = 4,
        N_HEXAHEDRON_NODE = 8,
        N_PRISM_NODE = 6,
        N_PYRAMID_NODE = 5,
    };

    struct SimplestGeometry {
        /// The vertex array of this element.
        std::vector<int> vertex;
    };

public:
    /// Default constructor.
    IOMesh() = default;
    /// Destructor.
    virtual ~IOMesh() = default;

    Mesh<DIM, DOW> create_mesh();

protected:
    /// Point array of the mesh.
    std::vector<point_t> pnt;
    /// Element array of the mesh.
    std::vector<SimplestGeometry> ele;
    /// Geometries arrays of the mesh.
    /// The geometries in \p n dimension are in the \p n-th entry of the array,
    /// which is still an array. */
    std::vector<GeometryBM> geo[dim+1];

protected:
    /// Number of points in the mesh.
    [[nodiscard]] inline int n_point() const { return pnt.size(); }
    /// Number of elements in the mesh.
    [[nodiscard]] inline int n_element() const { return ele.size(); }
    /// Point array.
    inline const std::vector<point_t> &point() const { return pnt; }
    /// Point array.
    inline std::vector<point_t> &point() { return pnt; }
    /// Certain point.
    inline const point_t &point(int i) const { return pnt[i]; }
    /// Certain point.
    inline point_t &point(int i) { return pnt[i]; }

    /// Number of geometries in certain dimension.
    [[nodiscard]] inline int n_geometry(int n) const {
        return geo[n].size();
    }
    /// Geometries array in certain dimension.
    [[nodiscard]] inline const std::vector<GeometryBM> &geometry(int n) const {
        return geo[n];
    }
    /// Geometries array in certain dimension.
    inline std::vector<GeometryBM> &geometry(int n) {
        return geo[n];
    }
    /// Certain geometry in certain dimension.
    [[nodiscard]] inline const GeometryBM &geometry(int i, int j) const {
        return geo[i][j];
    }
    /// Certain geometry in certain dimension.
    inline GeometryBM &geometry(int i, int j) {
        return geo[i][j];
    }
    /// Boundary marker of certain geometry in certain dimension.
    [[nodiscard]] inline int boundaryMark(int i, int j) const {
        return geo[i][j].bm;
    }
    /// Boundary marker of certain geometry in certain dimension.
    inline int &boundaryMark(int i, int j) {
        return geo[i][j].bm;
    }

    /// Element array.
    inline const std::vector<SimplestGeometry> &element() const { return ele; }
    /// Element array.
    inline std::vector<SimplestGeometry> &element() { return ele; }
    /// Certain element.
    inline const SimplestGeometry &element(int i) const { return ele[i]; }
    /// Certain element.
    inline SimplestGeometry &element(int i) { return ele[i]; }
    /// Vertex array of certain element.
    [[nodiscard]] inline const std::vector<int> &element_vertex(int i) const { return ele[i].vertex; }
    /// Vertex array of certain element.
    inline std::vector<int> &element_vertex(int i) { return ele[i].vertex; }
    /// Certain vertex of certain element.
    [[nodiscard]] inline int element_vertex(int i, int j) const { return ele[i].vertex[j]; }
    /// Certain vertex of certain element.
    inline int &element_vertex(int i, int j) { return ele[i].vertex[j]; }
};

}// namespace wp::fields
