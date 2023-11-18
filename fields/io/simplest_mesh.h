//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vector>
#include "core/vec.h"

namespace wp::fields {
/**
 * Simplest mesh is a kind of mesh with only the points coordinate and the element
 * geometry information. Some grid generation program provide such kind of data format.
 * This class provides facilities to generate a mesh with internal data format from a
 * simplest mesh. This class will be helpful when the data provided by grid generation
 * program. Generally, a grid generation program will be sure to provide this class
 * required. Warning: to generate a mesh with internal data format will be really
 * time-consuming.
 */
template<int DIM, int DOW = DIM>
class SimplestMesh {
public:
    using point_t = vec_t<DOW, float>;
    static constexpr uint32_t dim = DIM;
    static constexpr uint32_t dow = DOW;

    struct SimplestGeometry {
    public:
        /**< The vertex array of this element. */
        std::vector<int> vertex;
    };

protected:
    /**< Point array of the mesh. */
    std::vector<point_t> pnt;
    /**< Element array of the mesh. */
    std::vector<SimplestGeometry> ele;

public:
    /**< Default constructor. */
    SimplestMesh() = default;
    /**< Destructor. */
    virtual ~SimplestMesh() = default;

public:
    /**< Number of points in the mesh. */
    [[nodiscard]] int n_point() const { return pnt.size(); }
    /**< Number of elements in the mesh. */
    [[nodiscard]] int n_element() const { return ele.size(); }
    /**< Point array. */
    const std::vector<point_t> &point() const { return pnt; }
    /**< Point array. */
    std::vector<point_t> &point() { return pnt; }
    /**< Certain point. */
    const point_t &point(int i) const { return pnt[i]; }
    /**< Certain point. */
    point_t &point(int i) { return pnt[i]; }
    /**< Element array. */
    const std::vector<SimplestGeometry> &element() const { return ele; }
    /**< Element array. */
    std::vector<SimplestGeometry> &element() { return ele; }
    /**< Certain element. */
    const SimplestGeometry &element(int i) const { return ele[i]; }
    /**< Certain element. */
    SimplestGeometry &element(int i) { return ele[i]; }
    /**< Vertex array of certain element. */
    [[nodiscard]] const std::vector<int> &elementVertex(int i) const { return ele[i].vertex; }
    /**< Vertex array of certain element. */
    std::vector<int> &elementVertex(int i) { return ele[i].vertex; }
    /**< Certain vertex of certain element. */
    [[nodiscard]] int elementVertex(int i, int j) const { return ele[i].vertex[j]; }
    /**< Certain vertex of certain element. */
    int &elementVertex(int i, int j) { return ele[i].vertex[j]; }
};

}// namespace wp::fields
