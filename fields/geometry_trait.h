//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace wp::fields {
struct Interval;
struct IntervalTo2D;
struct Triangle;
struct TriangleTo3D;
struct TwinTriangle;
struct TwinTriangleTo3D;
struct Oblong;
struct OblongTo3D;
struct Tetrahedron;
struct TwinTetrahedron;
struct FourTetrahedron;
struct Recthexa;

struct Interval {
    static constexpr uint32_t dim = 1;
    static constexpr uint32_t tdim = 1;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;
    static constexpr uint32_t arr_len = 2;

    CUDA_CALLABLE static float volume(const point_t *v) {
        return (v[1][0] - v[0][0]);
    }

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

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        return (gv[1][0] - gv[0][0]) / (lv[1][0] - lv[0][0]);
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        return (lv[1][0] - lv[0][0]) / (gv[1][0] - gv[0][0]);
    }
};

struct IntervalTo2D {
    static constexpr uint32_t dim = 1;
    static constexpr uint32_t tdim = 2;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;
    using associate_t = Interval;
    static constexpr uint32_t arr_len = 2;

    CUDA_CALLABLE static float volume(const point_t *v) {
        return (v[1][0] - v[0][0]);
    }

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

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float gl = sqrt((gv[1][0] - gv[0][0]) * (gv[1][0] - gv[0][0]) + (gv[1][1] - gv[0][1]) * (gv[1][1] - gv[0][1]));
        return gl / fabs(lv[1][0] - lv[0][0]);
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float gl = sqrt((gv[1][0] - gv[0][0]) * (gv[1][0] - gv[0][0]) + (gv[1][1] - gv[0][1]) * (gv[1][1] - gv[0][1]));
        return fabs(lv[1][0] - lv[0][0]) / gl;
    }
};

struct Triangle {
    static constexpr uint32_t dim = 2;
    static constexpr uint32_t tdim = 2;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;
    static constexpr uint32_t arr_len = 3;

    CUDA_CALLABLE static constexpr float area(point_t a, point_t b, point_t c) {
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    }

    CUDA_CALLABLE static float volume(const point_t *v) {
        return .5f * ((v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[1][1] - v[0][1]) * (v[2][0] - v[0][0]));
    };

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

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float larea = Triangle::area(lv[0], lv[1], lv[2]);
        float garea = Triangle::area(gv[0], gv[1], gv[2]);
        return garea / larea;
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
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
    using associate_t = Triangle;
    static constexpr uint32_t arr_len = 3;

    CUDA_CALLABLE static float volume(const point_t *v) {
        return .5f * ((v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[1][1] - v[0][1]) * (v[2][0] - v[0][0]));
    };

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

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float larea = fabs(Triangle::area(lv[0], lv[1], lv[2]));
        float n[3];
        n[0] = ((gv[1][1] - gv[0][1]) * (gv[2][2] - gv[0][2]) - (gv[1][2] - gv[0][2]) * (gv[2][1] - gv[0][1]));
        n[1] = ((gv[1][2] - gv[0][2]) * (gv[2][0] - gv[0][0]) - (gv[1][0] - gv[0][0]) * (gv[2][2] - gv[0][2]));
        n[2] = ((gv[1][0] - gv[0][0]) * (gv[2][1] - gv[0][1]) - (gv[1][1] - gv[0][1]) * (gv[2][0] - gv[0][0]));
        float garea = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        return garea / larea;
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float larea = fabs(Triangle::area(lv[0], lv[1], lv[2]));
        float n[3];
        n[0] = ((gv[1][1] - gv[0][1]) * (gv[2][2] - gv[0][2]) - (gv[1][2] - gv[0][2]) * (gv[2][1] - gv[0][1]));
        n[1] = ((gv[1][2] - gv[0][2]) * (gv[2][0] - gv[0][0]) - (gv[1][0] - gv[0][0]) * (gv[2][2] - gv[0][2]));
        n[2] = ((gv[1][0] - gv[0][0]) * (gv[2][1] - gv[0][1]) - (gv[1][1] - gv[0][1]) * (gv[2][0] - gv[0][0]));
        float garea = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        return larea / garea;
    }
};

struct TwinTriangle {
    static constexpr uint32_t dim = 2;
    static constexpr uint32_t tdim = 2;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;
    static constexpr uint32_t arr_len = 4;

    CUDA_CALLABLE static float volume(const point_t *v) {
        return .5f * ((v[1][0] - v[0][0]) * (v[3][1] - v[0][1]) -
                      (v[3][0] - v[0][0]) * (v[1][1] - v[0][1]));
    }

    CUDA_CALLABLE static ref_point_t local_to_global(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        static int _[6] = {0, 1, 2, 0, 2, 3};
        float lambda[3], chi = Triangle::area(lp, lv[0], lv[2]);
        int I = (chi > 0) ? 3 : 0;
        float area = Triangle::area(lv[_[I]], lv[_[I + 1]], lv[_[I + 2]]);
        lambda[0] = Triangle::area(lp, lv[_[I + 1]], lv[_[I + 2]]) / area;
        lambda[1] = Triangle::area(lp, lv[_[I + 2]], lv[_[I]]) / area;
        lambda[2] = Triangle::area(lp, lv[_[I]], lv[_[I + 1]]) / area;
        return ref_point_t{lambda[0] * gv[_[I]][0] + lambda[1] * gv[_[I + 1]][0] + lambda[2] * gv[_[I + 2]][0],
                           lambda[0] * gv[_[I]][1] + lambda[1] * gv[_[I + 1]][1] + lambda[2] * gv[_[I + 2]][1]};
    }

    CUDA_CALLABLE static point_t global_to_local(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        static int _[6] = {0, 1, 2, 0, 2, 3};
        float lambda[3], chi = Triangle::area(gp, gv[0], gv[2]);
        int I = (chi > 0) ? 3 : 0;
        float area = Triangle::area(gv[_[I]], gv[_[I + 1]], gv[_[I + 2]]);
        lambda[0] = Triangle::area(gp, gv[_[I + 1]], gv[_[I + 2]]) / area;
        lambda[1] = Triangle::area(gp, gv[_[I + 2]], gv[_[I]]) / area;
        lambda[2] = Triangle::area(gp, gv[_[I]], gv[_[I + 1]]) / area;
        return point_t{lambda[0] * lv[_[I]][0] + lambda[1] * lv[_[I + 1]][0] + lambda[2] * lv[_[I + 2]][0],
                       lambda[0] * lv[_[I]][1] + lambda[1] * lv[_[I + 1]][1] + lambda[2] * lv[_[I + 2]][1]};
    }

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        static int _[6] = {0, 1, 2, 0, 2, 3};
        float chi = Triangle::area(lp, lv[0], lv[2]);
        int I = (chi > 0) ? 3 : 0;
        float larea = Triangle::area(lv[_[I]], lv[_[I + 1]], lv[_[I + 2]]);
        float garea = Triangle::area(gv[_[I]], gv[_[I + 1]], gv[_[I + 2]]);
        return garea / larea;
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        static int _[6] = {0, 1, 2, 0, 2, 3};
        float chi = Triangle::area(gp, gv[0], gv[2]);
        int I = (chi > 0) ? 3 : 0;
        float larea = Triangle::area(lv[_[I]], lv[_[I + 1]], lv[_[I + 2]]);
        float garea = Triangle::area(gv[_[I]], gv[_[I + 1]], gv[_[I + 2]]);
        return larea / garea;
    }
};

struct TwinTriangleTo3D {
    static constexpr uint32_t dim = 2;
    static constexpr uint32_t tdim = 3;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;
    using associate_t = TwinTriangle;
    static constexpr uint32_t arr_len = 4;

    CUDA_CALLABLE static float volume(const point_t *v) {
        return .5f * ((v[1][0] - v[0][0]) * (v[3][1] - v[0][1]) -
                      (v[3][0] - v[0][0]) * (v[1][1] - v[0][1]));
    }

    CUDA_CALLABLE static ref_point_t local_to_global(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float lambda[3];
        float area = Triangle::area(lv[0], lv[1], lv[3]);
        lambda[0] = Triangle::area(lp, lv[1], lv[3]) / area;
        lambda[1] = Triangle::area(lp, lv[3], lv[0]) / area;
        lambda[2] = Triangle::area(lp, lv[0], lv[1]) / area;
        return ref_point_t{lambda[0] * gv[0][0] + lambda[1] * gv[1][0] + lambda[2] * gv[3][0],
                           lambda[0] * gv[0][1] + lambda[1] * gv[1][1] + lambda[2] * gv[3][1],
                           lambda[0] * gv[0][2] + lambda[1] * gv[1][2] + lambda[2] * gv[3][2]};
    }

    CUDA_CALLABLE static point_t global_to_local(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        // todo
        return point_t{0, 0};
    }

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float larea = fabs(Triangle::area(lv[0], lv[1], lv[3]));
        float n[3];
        n[0] = ((gv[1][1] - gv[0][1]) * (gv[3][2] - gv[0][2]) - (gv[1][2] - gv[0][2]) * (gv[3][1] - gv[0][1]));
        n[1] = ((gv[1][2] - gv[0][2]) * (gv[3][0] - gv[0][0]) - (gv[1][0] - gv[0][0]) * (gv[3][2] - gv[0][2]));
        n[2] = ((gv[1][0] - gv[0][0]) * (gv[3][1] - gv[0][1]) - (gv[1][1] - gv[0][1]) * (gv[3][0] - gv[0][0]));
        float garea = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        return garea / larea;
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        // todo
        return 0.0;
    }
};

struct Oblong {
    static constexpr uint32_t dim = 2;
    static constexpr uint32_t tdim = 2;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;
    static constexpr uint32_t arr_len = 4;

#define COPY_TO_A                  \
    a[0][0] = gv[1][0] - gv[0][0]; \
    a[0][1] = gv[1][1] - gv[0][1]; \
    a[1][0] = gv[3][0] - gv[0][0]; \
    a[1][1] = gv[3][1] - gv[0][1];

    CUDA_CALLABLE static float det(const point_t v1, const point_t v2) { return v1[0] * v2[1] - v1[1] * v2[0]; }

    CUDA_CALLABLE static float volume(const point_t *v) {
        return 1.f;
    };

    CUDA_CALLABLE static ref_point_t local_to_global(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float a[3];
        a[0] = .5f * (-lp[0] - lp[1]);
        a[1] = .5f * (1.f + lp[0]);
        a[2] = .5f * (1.f + lp[1]);

        return ref_point_t{a[0] * gv[0][0] + a[1] * gv[1][0] + a[2] * gv[3][0],
                           a[0] * gv[0][1] + a[1] * gv[1][1] + a[2] * gv[3][1]};
    }

    CUDA_CALLABLE static point_t global_to_local(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float det0;
        point_t a[3];
        COPY_TO_A

        a[2][0] = 2.f * gp[0] - (gv[1][0] + gv[3][0]);
        a[2][1] = 2.f * gp[1] - (gv[1][1] + gv[3][1]);

        det0 = det(a[0], a[1]);
        return point_t{det(a[2], a[1]) / det0,
                       det(a[0], a[2]) / det0};
    }

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        point_t a[2];
        COPY_TO_A

        return det(a[0], a[1]) / 4.f;
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        point_t a[2];
        COPY_TO_A
        return 4.f / det(a[0], a[1]);
    }
#undef COPY_TO_A
};

struct OblongTo3D {
    static constexpr uint32_t dim = 2;
    static constexpr uint32_t tdim = 3;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;
    using associate_t = Oblong;
    static constexpr uint32_t arr_len = 4;

    CUDA_CALLABLE static float volume(const point_t *v) {
        return 1.f;
    };

    CUDA_CALLABLE static ref_point_t local_to_global(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float a[3];
        a[0] = .5f * (-lp[0] - lp[1]);
        a[1] = .5f * (1.f + lp[0]);
        a[2] = .5f * (1.f + lp[1]);

        return ref_point_t{a[0] * gv[0][0] + a[1] * gv[1][0] + a[2] * gv[3][0],
                           a[0] * gv[0][1] + a[1] * gv[1][1] + a[2] * gv[3][1],
                           a[0] * gv[0][2] + a[1] * gv[1][2] + a[2] * gv[3][2]};
    }

    CUDA_CALLABLE static point_t global_to_local(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        // todo
        return point_t{0, 0};
    }

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        ref_point_t a[2];
        a[0][0] = gv[1][0] - gv[0][0];
        a[0][1] = gv[1][1] - gv[0][1];
        a[0][2] = gv[1][2] - gv[0][2];
        a[1][0] = gv[3][0] - gv[0][0];
        a[1][1] = gv[3][1] - gv[0][1];
        a[1][2] = gv[3][2] - gv[0][2];

        ref_point_t n;
        n[0] = a[0][1] * a[1][2] - a[0][2] * a[1][1];
        n[1] = -a[0][0] * a[1][2] - a[0][2] * a[1][0];
        n[2] = a[0][0] * a[1][1] - a[0][1] * a[1][0];

        return sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) / 4.f;
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        ref_point_t a[2];
        a[0][0] = gv[1][0] - gv[0][0];
        a[0][1] = gv[1][1] - gv[0][1];
        a[0][2] = gv[1][2] - gv[0][2];
        a[1][0] = gv[3][0] - gv[0][0];
        a[1][1] = gv[3][1] - gv[0][1];
        a[1][2] = gv[3][2] - gv[0][2];

        ref_point_t n;
        n[0] = a[0][1] * a[1][2] - a[0][2] * a[1][1];
        n[1] = -a[0][0] * a[1][2] - a[0][2] * a[1][0];
        n[2] = a[0][0] * a[1][1] - a[0][1] * a[1][0];

        return 4.f / sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    }
};

struct Tetrahedron {
    static constexpr uint32_t dim = 3;
    static constexpr uint32_t tdim = 3;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;
    static constexpr uint32_t arr_len = 4;

    CUDA_CALLABLE static float volume(const point_t *v) {
        return ((v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) * (v[3][2] - v[0][2]) + (v[1][1] - v[0][1]) * (v[2][2] - v[0][2]) * (v[3][0] - v[0][0]) + (v[1][2] - v[0][2]) * (v[2][0] - v[0][0]) * (v[3][1] - v[0][1]) - (v[1][0] - v[0][0]) * (v[2][2] - v[0][2]) * (v[3][1] - v[0][1]) - (v[1][1] - v[0][1]) * (v[2][0] - v[0][0]) * (v[3][2] - v[0][2]) - (v[1][2] - v[0][2]) * (v[2][1] - v[0][1]) * (v[3][0] - v[0][0])) / 6.f;
    };

    CUDA_CALLABLE static float get_volume(const point_t v0, const point_t v1, const point_t v2, const point_t v3) {
        return ((v1[0] - v0[0]) * (v2[1] - v0[1]) * (v3[2] - v0[2]) + (v1[1] - v0[1]) * (v2[2] - v0[2]) * (v3[0] - v0[0]) +
                (v1[2] - v0[2]) * (v2[0] - v0[0]) * (v3[1] - v0[1]) - (v1[0] - v0[0]) * (v2[2] - v0[2]) * (v3[1] - v0[1]) -
                (v1[1] - v0[1]) * (v2[0] - v0[0]) * (v3[2] - v0[2]) - (v1[2] - v0[2]) * (v2[1] - v0[1]) * (v3[0] - v0[0])) /
               6.f;
    }

    CUDA_CALLABLE static ref_point_t local_to_global(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float lambda[4];
        float volume = get_volume(lv[0], lv[1], lv[2], lv[3]);
        lambda[0] = get_volume(lp, lv[1], lv[2], lv[3]) / volume;
        lambda[1] = get_volume(lv[0], lp, lv[2], lv[3]) / volume;
        lambda[2] = get_volume(lv[0], lv[1], lp, lv[3]) / volume;
        lambda[3] = get_volume(lv[0], lv[1], lv[2], lp) / volume;
        return ref_point_t{lambda[0] * gv[0][0] + lambda[1] * gv[1][0] + lambda[2] * gv[2][0] + lambda[3] * gv[3][0],
                           lambda[0] * gv[0][1] + lambda[1] * gv[1][1] + lambda[2] * gv[2][1] + lambda[3] * gv[3][1],
                           lambda[0] * gv[0][2] + lambda[1] * gv[1][2] + lambda[2] * gv[2][2] + lambda[3] * gv[3][2]};
    }

    CUDA_CALLABLE static point_t global_to_local(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float lambda[4];
        float volume = get_volume(gv[0], gv[1], gv[2], gv[3]);
        lambda[0] = get_volume(gp, gv[1], gv[2], gv[3]) / volume;
        lambda[1] = get_volume(gv[0], gp, gv[2], gv[3]) / volume;
        lambda[2] = get_volume(gv[0], gv[1], gp, gv[3]) / volume;
        lambda[3] = get_volume(gv[0], gv[1], gv[2], gp) / volume;
        return point_t{lambda[0] * lv[0][0] + lambda[1] * lv[1][0] + lambda[2] * lv[2][0] + lambda[3] * lv[3][0],
                       lambda[0] * lv[0][1] + lambda[1] * lv[1][1] + lambda[2] * lv[2][1] + lambda[3] * lv[3][1],
                       lambda[0] * lv[0][2] + lambda[1] * lv[1][2] + lambda[2] * lv[2][2] + lambda[3] * lv[3][2]};
    }

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float lvolume = get_volume(lv[0], lv[1], lv[2], lv[3]);
        float gvolume = get_volume(gv[0], gv[1], gv[2], gv[3]);
        return gvolume / lvolume;
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float lvolume = get_volume(lv[0], lv[1], lv[2], lv[3]);
        float gvolume = get_volume(gv[0], gv[1], gv[2], gv[3]);
        return lvolume / gvolume;
    }
};

struct TwinTetrahedron {
    static constexpr uint32_t dim = 3;
    static constexpr uint32_t tdim = 3;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;
    static constexpr uint32_t arr_len = 5;

    CUDA_CALLABLE static float volume(const point_t *v) {
        return ((v[1][0] - v[0][0]) * (v[3][1] - v[0][1]) * (v[4][2] - v[0][2]) + (v[1][1] - v[0][1]) * (v[3][2] - v[0][2]) * (v[4][0] - v[0][0]) + (v[1][2] - v[0][2]) * (v[3][0] - v[0][0]) * (v[4][1] - v[0][1]) - (v[1][0] - v[0][0]) * (v[3][2] - v[0][2]) * (v[4][1] - v[0][1]) - (v[1][1] - v[0][1]) * (v[3][0] - v[0][0]) * (v[4][2] - v[0][2]) - (v[1][2] - v[0][2]) * (v[3][1] - v[0][1]) * (v[4][0] - v[0][0])) / 6.f;
    };

    CUDA_CALLABLE static ref_point_t local_to_global(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float lambda[4];
        float volume = Tetrahedron::get_volume(lv[0], lv[1], lv[3], lv[4]);
        lambda[0] = Tetrahedron::get_volume(lp, lv[1], lv[3], lv[4]) / volume;
        lambda[1] = Tetrahedron::get_volume(lv[0], lp, lv[3], lv[4]) / volume;
        lambda[2] = Tetrahedron::get_volume(lv[0], lv[1], lp, lv[4]) / volume;
        lambda[3] = Tetrahedron::get_volume(lv[0], lv[1], lv[3], lp) / volume;
        return ref_point_t{lambda[0] * gv[0][0] + lambda[1] * gv[1][0] + lambda[2] * gv[3][0] + lambda[3] * gv[4][0],
                           lambda[0] * gv[0][1] + lambda[1] * gv[1][1] + lambda[2] * gv[3][1] + lambda[3] * gv[4][1],
                           lambda[0] * gv[0][2] + lambda[1] * gv[1][2] + lambda[2] * gv[3][2] + lambda[3] * gv[4][2]};
    }

    CUDA_CALLABLE static point_t global_to_local(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float lambda[4];
        float volume = Tetrahedron::get_volume(gv[0], gv[1], gv[3], gv[4]);
        lambda[0] = Tetrahedron::get_volume(gp, gv[1], gv[3], gv[4]) / volume;
        lambda[1] = Tetrahedron::get_volume(gv[0], gp, gv[3], gv[4]) / volume;
        lambda[2] = Tetrahedron::get_volume(gv[0], gv[1], gp, gv[4]) / volume;
        lambda[3] = Tetrahedron::get_volume(gv[0], gv[1], gv[3], gp) / volume;
        return point_t{lambda[0] * lv[0][0] + lambda[1] * lv[1][0] + lambda[2] * lv[3][0] + lambda[3] * lv[4][0],
                       lambda[0] * lv[0][1] + lambda[1] * lv[1][1] + lambda[2] * lv[3][1] + lambda[3] * lv[4][1],
                       lambda[0] * lv[0][2] + lambda[1] * lv[1][2] + lambda[2] * lv[3][2] + lambda[3] * lv[4][2]};
    }

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float lvolume = Tetrahedron::get_volume(lv[0], lv[1], lv[3], lv[4]);
        float gvolume = Tetrahedron::get_volume(gv[0], gv[1], gv[3], gv[4]);
        return gvolume / lvolume;
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float lvolume = Tetrahedron::get_volume(lv[0], lv[1], lv[3], lv[4]);
        float gvolume = Tetrahedron::get_volume(gv[0], gv[1], gv[3], gv[4]);
        return lvolume / gvolume;
    }
};

struct FourTetrahedron {
    static constexpr uint32_t dim = 3;
    static constexpr uint32_t tdim = 3;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;
    static constexpr uint32_t arr_len = 4;

    CUDA_CALLABLE static float volume(const point_t *v) {
        return ((v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) * (v[3][2] - v[0][2]) + (v[1][1] - v[0][1]) * (v[2][2] - v[0][2]) * (v[3][0] - v[0][0]) + (v[1][2] - v[0][2]) * (v[2][0] - v[0][0]) * (v[3][1] - v[0][1]) - (v[1][0] - v[0][0]) * (v[2][2] - v[0][2]) * (v[3][1] - v[0][1]) - (v[1][1] - v[0][1]) * (v[2][0] - v[0][0]) * (v[3][2] - v[0][2]) - (v[1][2] - v[0][2]) * (v[2][1] - v[0][1]) * (v[3][0] - v[0][0])) / 6.f;
    };

    CUDA_CALLABLE static ref_point_t local_to_global(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float lambda[4];
        float volume = Tetrahedron::get_volume(lv[0], lv[1], lv[2], lv[3]);
        lambda[0] = Tetrahedron::get_volume(lp, lv[1], lv[2], lv[3]) / volume;
        lambda[1] = Tetrahedron::get_volume(lv[0], lp, lv[2], lv[3]) / volume;
        lambda[2] = Tetrahedron::get_volume(lv[0], lv[1], lp, lv[3]) / volume;
        lambda[3] = Tetrahedron::get_volume(lv[0], lv[1], lv[2], lp) / volume;
        return ref_point_t{lambda[0] * gv[0][0] + lambda[1] * gv[1][0] + lambda[2] * gv[2][0] + lambda[3] * gv[3][0],
                           lambda[0] * gv[0][1] + lambda[1] * gv[1][1] + lambda[2] * gv[2][1] + lambda[3] * gv[3][1],
                           lambda[0] * gv[0][2] + lambda[1] * gv[1][2] + lambda[2] * gv[2][2] + lambda[3] * gv[3][2]};
    }

    CUDA_CALLABLE static point_t global_to_local(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float lambda[4];
        float volume = Tetrahedron::get_volume(gv[0], gv[1], gv[2], gv[3]);
        lambda[0] = Tetrahedron::get_volume(gp, gv[1], gv[2], gv[3]) / volume;
        lambda[1] = Tetrahedron::get_volume(gv[0], gp, gv[2], gv[3]) / volume;
        lambda[2] = Tetrahedron::get_volume(gv[0], gv[1], gp, gv[3]) / volume;
        lambda[3] = Tetrahedron::get_volume(gv[0], gv[1], gv[2], gp) / volume;
        return point_t{lambda[0] * lv[0][0] + lambda[1] * lv[1][0] + lambda[2] * lv[2][0] + lambda[3] * lv[3][0],
                       lambda[0] * lv[0][1] + lambda[1] * lv[1][1] + lambda[2] * lv[2][1] + lambda[3] * lv[3][1],
                       lambda[0] * lv[0][2] + lambda[1] * lv[1][2] + lambda[2] * lv[2][2] + lambda[3] * lv[3][2]};
    }

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float lvolume = Tetrahedron::get_volume(lv[0], lv[1], lv[2], lv[3]);
        float gvolume = Tetrahedron::get_volume(gv[0], gv[1], gv[2], gv[3]);
        return gvolume / lvolume;
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float lvolume = Tetrahedron::get_volume(lv[0], lv[1], lv[2], lv[3]);
        float gvolume = Tetrahedron::get_volume(gv[0], gv[1], gv[2], gv[3]);
        return lvolume / gvolume;
    }
};

struct Recthexa {
    static constexpr uint32_t dim = 3;
    static constexpr uint32_t tdim = 3;
    using point_t = vec_t<dim, float>;
    using ref_point_t = vec_t<tdim, float>;
    static constexpr uint32_t arr_len = 5;

#define COPY_TO_A                  \
    a[0][0] = gv[1][0] - gv[0][0]; \
    a[0][1] = gv[1][1] - gv[0][1]; \
    a[0][2] = gv[1][2] - gv[0][2]; \
    a[1][0] = gv[3][0] - gv[0][0]; \
    a[1][1] = gv[3][1] - gv[0][1]; \
    a[1][2] = gv[3][2] - gv[0][2]; \
    a[2][0] = gv[4][0] - gv[0][0]; \
    a[2][1] = gv[4][1] - gv[0][1]; \
    a[2][2] = gv[4][2] - gv[0][2];

    CUDA_CALLABLE static float det(const point_t v1, const point_t v2, const point_t v3) {
        return v1[0] * v2[1] * v3[2] + v1[1] * v2[2] * v3[0] + v1[2] * v2[0] * v3[1] - v1[0] * v2[2] * v3[1] -
               v1[1] * v2[0] * v3[2] - v1[2] * v2[1] * v3[0];
    }

    CUDA_CALLABLE static float volume(const point_t *v) {
        return 1.;
    };

    CUDA_CALLABLE static ref_point_t local_to_global(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        float a[4];
        a[0] = .5f * (-1.f - lp[0] - lp[1] - lp[2]);
        a[1] = .5f * (1.f + lp[0]);
        a[2] = .5f * (1.f + lp[1]);
        a[3] = .5f * (1.f + lp[2]);

        return ref_point_t{a[0] * gv[0][0] + a[1] * gv[1][0] + a[2] * gv[3][0] + a[3] * gv[4][0],
                           a[0] * gv[0][1] + a[1] * gv[1][1] + a[2] * gv[3][1] + a[3] * gv[4][1],
                           a[0] * gv[0][2] + a[1] * gv[1][2] + a[2] * gv[3][2] + a[3] * gv[4][2]};
    }

    CUDA_CALLABLE static point_t global_to_local(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        float det0;
        point_t a[4];
        COPY_TO_A

        a[3][0] = 2.f * gp[0] - (gv[1][0] + gv[3][0] + gv[4][0] - gv[0][0]);
        a[3][1] = 2.f * gp[1] - (gv[1][1] + gv[3][1] + gv[4][1] - gv[0][1]);
        a[3][2] = 2.f * gp[2] - (gv[1][2] + gv[3][2] + gv[4][2] - gv[0][2]);

        det0 = det(a[0], a[1], a[2]);
        return point_t{det(a[3], a[1], a[2]) / det0,
                       det(a[0], a[3], a[2]) / det0,
                       det(a[0], a[1], a[3]) / det0};
    }

    CUDA_CALLABLE static float local_to_global_jacobian(const point_t lp, const point_t *lv, const ref_point_t *gv) {
        point_t a[3];
        COPY_TO_A

        return det(a[0], a[1], a[2]) / 8.f;
    }

    CUDA_CALLABLE static float global_to_local_jacobian(const ref_point_t gp, const point_t *lv, const ref_point_t *gv) {
        point_t a[3];
        COPY_TO_A
        return 8.f / det(a[0], a[1], a[2]);
    }
#undef COPY_TO_A
};

}// namespace wp::fields
