//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace wp::fields {
struct Interval {
    static constexpr uint32_t dim = 1;
    static constexpr uint32_t tdim = 1;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;

    CUDA_CALLABLE static ref_point_t local_to_global(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float lambda[2];
        lambda[0] = (lv[1][0] - lp[0]) / (lv[1][0] - lv[0][0]);
        lambda[1] = (lp[0] - lv[0][0]) / (lv[1][0] - lv[0][0]);
        return point_t{lambda[0] * gv[0][0] + lambda[1] * gv[1][0]};
    }

    CUDA_CALLABLE static point_t global_to_local(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float lambda[2];
        lambda[0] = (gv[1][0] - gp[0]) / (gv[1][0] - gv[0][0]);
        lambda[1] = (gp[0] - gv[0][0]) / (gv[1][0] - gv[0][0]);
        return point_t{lambda[0] * lv[0][0] + lambda[1] * lv[1][0]};
    }

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t *lv, const ref_point_t *gv) {
        return (gv[1][0] - gv[0][0]) / (lv[1][0] - lv[0][0]);
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const point_t *lv, const ref_point_t *gv) {
        return (lv[1][0] - lv[0][0]) / (gv[1][0] - gv[0][0]);
    }
};

struct IntervalTo2D {
    static constexpr uint32_t dim = 1;
    static constexpr uint32_t tdim = 2;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;

    CUDA_CALLABLE static ref_point_t local_to_global(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float lambda[2];
        lambda[0] = (lv[1][0] - lp[0]) / (lv[1][0] - lv[0][0]);
        lambda[1] = (lp[0] - lv[0][0]) / (lv[1][0] - lv[0][0]);
        return ref_point_t{lambda[0] * gv[0][0] + lambda[1] * gv[1][0],
                           lambda[0] * gv[0][1] + lambda[1] * gv[1][1]};
    }

    CUDA_CALLABLE static point_t global_to_local(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float lambda[2];
        lambda[1] = sqrt((gv[1][0] - gv[0][0]) * (gv[1][0] - gv[0][0]) + (gv[1][1] - gv[0][1]) * (gv[1][1] - gv[0][1]));
        lambda[0] = sqrt((gv[1][0] - gp[0]) * (gv[1][0] - gp[0]) + (gv[1][1] - gp[1]) * (gv[1][1] - gp[1])) / lambda[1];
        lambda[1] = 1 - lambda[0];
        return point_t{lambda[0] * lv[0][0] + lambda[1] * lv[1][0]};
    }

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t *lv, const ref_point_t *gv) {
        float gl = sqrt((gv[1][0] - gv[0][0]) * (gv[1][0] - gv[0][0]) + (gv[1][1] - gv[0][1]) * (gv[1][1] - gv[0][1]));
        return gl / fabs(lv[1][0] - lv[0][0]);
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const point_t *lv, const ref_point_t *gv) {
        float gl = sqrt((gv[1][0] - gv[0][0]) * (gv[1][0] - gv[0][0]) + (gv[1][1] - gv[0][1]) * (gv[1][1] - gv[0][1]));
        return fabs(lv[1][0] - lv[0][0]) / gl;
    }
};

struct Triangle {
    static constexpr uint32_t dim = 2;
    static constexpr uint32_t tdim = 2;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;

    CUDA_CALLABLE static constexpr float area(point_t a, point_t b, point_t c) {
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    }

    CUDA_CALLABLE static ref_point_t local_to_global(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float lambda[3];
        float area = Triangle::area(lv[0], lv[1], lv[2]);
        lambda[0] = Triangle::area(lp, lv[1], lv[2]) / area;
        lambda[1] = Triangle::area(lp, lv[2], lv[0]) / area;
        lambda[2] = Triangle::area(lp, lv[0], lv[1]) / area;
        return ref_point_t{lambda[0] * gv[0][0] + lambda[1] * gv[1][0] + lambda[2] * gv[2][0],
                           lambda[0] * gv[0][1] + lambda[1] * gv[1][1] + lambda[2] * gv[2][1]};
    }

    CUDA_CALLABLE static point_t global_to_local(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float lambda[3];
        float area = Triangle::area(gv[0], gv[1], gv[2]);
        lambda[0] = Triangle::area(gp, gv[1], gv[2]) / area;
        lambda[1] = Triangle::area(gp, gv[2], gv[0]) / area;
        lambda[2] = Triangle::area(gp, gv[0], gv[1]) / area;
        return point_t{lambda[0] * lv[0][0] + lambda[1] * lv[1][0] + lambda[2] * lv[2][0],
                       lambda[0] * lv[0][1] + lambda[1] * lv[1][1] + lambda[2] * lv[2][1]};
    }

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t *lv, const ref_point_t *gv) {
        float larea = Triangle::area(lv[0], lv[1], lv[2]);
        float garea = Triangle::area(gv[0], gv[1], gv[2]);
        return garea / larea;
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const point_t *lv, const ref_point_t *gv) {
        float larea = Triangle::area(lv[0], lv[1], lv[2]);
        float garea = Triangle::area(gv[0], gv[1], gv[2]);
        return larea / garea;
    }
};

struct TriangleTo3D {
    static constexpr uint32_t dim = 2;
    static constexpr uint32_t tdim = 3;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;

    CUDA_CALLABLE static ref_point_t local_to_global(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float lambda[3];
        float area = Triangle::area(lv[0], lv[1], lv[2]);
        lambda[0] = Triangle::area(lp, lv[1], lv[2]) / area;
        lambda[1] = Triangle::area(lp, lv[2], lv[0]) / area;
        lambda[2] = Triangle::area(lp, lv[0], lv[1]) / area;
        return ref_point_t{lambda[0] * gv[0][0] + lambda[1] * gv[1][0] + lambda[2] * gv[2][0],
                           lambda[0] * gv[0][1] + lambda[1] * gv[1][1] + lambda[2] * gv[2][1],
                           lambda[0] * gv[0][2] + lambda[1] * gv[1][2] + lambda[2] * gv[2][2]};
    }

    CUDA_CALLABLE static point_t global_to_local(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float lambda[3];
        float n[3];
        n[0] = ((gv[1][1] - gv[0][1]) * (gv[2][2] - gv[0][2]) - (gv[1][2] - gv[0][2]) * (gv[2][1] - gv[0][1]));
        n[1] = ((gv[1][2] - gv[0][2]) * (gv[2][0] - gv[0][0]) - (gv[1][0] - gv[0][0]) * (gv[2][2] - gv[0][2]));
        n[2] = ((gv[1][0] - gv[0][0]) * (gv[2][1] - gv[0][1]) - (gv[1][1] - gv[0][1]) * (gv[2][0] - gv[0][0]));
        float area = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        n[0] = ((gv[1][1] - gp[1]) * (gv[2][2] - gp[2]) - (gv[1][2] - gp[2]) * (gv[2][1] - gp[1]));
        n[1] = ((gv[1][2] - gp[2]) * (gv[2][0] - gp[0]) - (gv[1][0] - gp[0]) * (gv[2][2] - gp[2]));
        n[2] = ((gv[1][0] - gp[0]) * (gv[2][1] - gp[1]) - (gv[1][1] - gp[1]) * (gv[2][0] - gp[0]));
        lambda[0] = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) / area;
        n[0] = ((gp[1] - gv[0][1]) * (gv[2][2] - gv[0][2]) - (gp[2] - gv[0][2]) * (gv[2][1] - gv[0][1]));
        n[1] = ((gp[2] - gv[0][2]) * (gv[2][0] - gv[0][0]) - (gp[0] - gv[0][0]) * (gv[2][2] - gv[0][2]));
        n[2] = ((gp[0] - gv[0][0]) * (gv[2][1] - gv[0][1]) - (gp[1] - gv[0][1]) * (gv[2][0] - gv[0][0]));
        lambda[1] = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) / area;
        n[0] = ((gv[1][1] - gv[0][1]) * (gp[2] - gv[0][2]) - (gv[1][2] - gv[0][2]) * (gp[1] - gv[0][1]));
        n[1] = ((gv[1][2] - gv[0][2]) * (gp[0] - gv[0][0]) - (gv[1][0] - gv[0][0]) * (gp[2] - gv[0][2]));
        n[2] = ((gv[1][0] - gv[0][0]) * (gp[1] - gv[0][1]) - (gv[1][1] - gv[0][1]) * (gp[0] - gv[0][0]));
        lambda[2] = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) / area;
        return point_t{lambda[0] * lv[0][0] + lambda[1] * lv[1][0] + lambda[2] * lv[2][0],
                       lambda[0] * lv[0][1] + lambda[1] * lv[1][1] + lambda[2] * lv[2][1]};
    }

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t *lv, const ref_point_t *gv) {
        float larea = fabs(Triangle::area(lv[0], lv[1], lv[2]));
        float n[3];
        n[0] = ((gv[1][1] - gv[0][1]) * (gv[2][2] - gv[0][2]) - (gv[1][2] - gv[0][2]) * (gv[2][1] - gv[0][1]));
        n[1] = ((gv[1][2] - gv[0][2]) * (gv[2][0] - gv[0][0]) - (gv[1][0] - gv[0][0]) * (gv[2][2] - gv[0][2]));
        n[2] = ((gv[1][0] - gv[0][0]) * (gv[2][1] - gv[0][1]) - (gv[1][1] - gv[0][1]) * (gv[2][0] - gv[0][0]));
        float garea = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        return garea / larea;
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const point_t *lv, const ref_point_t *gv) {
        float larea = fabs(Triangle::area(lv[0], lv[1], lv[2]));
        float n[3];
        n[0] = ((gv[1][1] - gv[0][1]) * (gv[2][2] - gv[0][2]) - (gv[1][2] - gv[0][2]) * (gv[2][1] - gv[0][1]));
        n[1] = ((gv[1][2] - gv[0][2]) * (gv[2][0] - gv[0][0]) - (gv[1][0] - gv[0][0]) * (gv[2][2] - gv[0][2]));
        n[2] = ((gv[1][0] - gv[0][0]) * (gv[2][1] - gv[0][1]) - (gv[1][1] - gv[0][1]) * (gv[2][0] - gv[0][0]));
        float garea = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        return larea / garea;
    }
};

}// namespace wp::fields
