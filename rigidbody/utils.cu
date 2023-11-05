//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "utils.h"

namespace wp {
CUDA_CALLABLE vec3f velocity_at_point(spatial_vectorf qd, vec3f r) {
    return cross(spatial_top(qd), r) + spatial_bottom(qd);
}

CUDA_CALLABLE quatf quat_twist(vec3f axis, quatf q) {
    auto a = vec3(q.x, q.y, q.z);
    auto proj = dot(a, axis);
    a = proj * axis;
    return normalize(quat(a[0], a[1], a[2], q.w));
}

CUDA_CALLABLE float quat_twist_angle(vec3f axis, quatf q) {
    return 2.f * acos(quat_twist(axis, q).w);
}

CUDA_CALLABLE vec3 quat_decompose(quatf q) {
    auto R = mat33(
        quat_rotate(q, vec3(1.0, 0.0, 0.0)),
        quat_rotate(q, vec3(0.0, 1.0, 0.0)),
        quat_rotate(q, vec3(0.0, 0.0, 1.0)));

    // https://www.sedris.org/wg8home/Documents/WG80485.pdf
    auto phi = atan2(R(1, 2), R(2, 2));
    auto sinp = -R(0, 2);
    float theta;
    if (abs(sinp) >= 1.0) {
        theta = 1.57079632679f * sign(sinp);
    } else {
        theta = asin(-R(0, 2));
    }
    auto psi = atan2(R(0, 1), R(0, 0));

    return -vec3(phi, theta, psi);
}

CUDA_CALLABLE vec3 quat_to_rpy(quatf q) {
    auto x = q.x;
    auto y = q.y;
    auto z = q.z;
    auto w = q.w;
    auto t0 = 2.f * (w * x + y * z);
    auto t1 = 1.f - 2.f * (x * x + y * y);
    auto roll_x = atan2(t0, t1);

    auto t2 = 2.f * (w * y - z * x);
    t2 = clamp(t2, -1.f, 1.f);
    auto pitch_y = asin(t2);

    auto t3 = 2.f * (w * z + x * y);
    auto t4 = 1.f - 2.f * (y * y + z * z);
    auto yaw_z = atan2(t3, t4);

    return {roll_x, pitch_y, yaw_z};
}

CUDA_CALLABLE vec3f quat_to_euler(quatf q, int i, int j, int k) {
    // https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276302
    bool not_proper = true;
    if (i == k) {
        not_proper = false;
        k = 6 - i - j;// because i + j + k = 1 + 2 + 3 = 6
    }
    auto e = float((i - j) * (j - k) * (k - i)) / 2.f;// Levi-Civita symbol
    auto a = q[0];
    auto b = q[i];
    auto c = q[j];
    auto d = q[k] * e;
    if (not_proper) {
        a -= q[j];
        b += q[k] * e;
        c += q[0];
        d -= q[i];
    }
    auto t2 = acos(2.f * (a * a + b * b) / (a * a + b * b + c * c + d * d) - 1.f);
    auto tp = atan2(b, a);
    auto tm = atan2(d, c);
    auto t1 = 0.f;
    auto t3 = 0.f;
    if (abs(t2) < 1e-6) {
        t3 = 2.f * tp - t1;
    } else if (abs(t2 - PI_2) < 1e-6) {
        t3 = 2.f * tm + t1;
    } else {
        t1 = tp - tm;
        t3 = tp + tm;
    }
    if (not_proper) {
        t2 -= PI_2;
        t3 *= e;
    }
    return {t1, t2, t3};
}

CUDA_CALLABLE quatf quat_between_vectors(vec3f a, vec3f b) {
    a = normalize(a);
    b = normalize(b);
    auto c = cross(a, b);
    auto d = dot(a, b);
    auto q = quat(c[0], c[1], c[2], 1.f + d);
    return normalize(q);
}

CUDA_CALLABLE spatial_vector transform_twist(transformf t, spatial_vectorf x) {
    //  Frank & Park definition 3.20, pg 100

    auto q = transform_get_rotation(t);
    auto p = transform_get_translation(t);

    auto w = spatial_top(x);
    auto v = spatial_bottom(x);

    w = quat_rotate(q, w);
    v = quat_rotate(q, v) + cross(p, w);

    return {w, v};
}

CUDA_CALLABLE spatial_vector transform_wrench(transformf t, spatial_vectorf x) {
    auto q = transform_get_rotation(t);
    auto p = transform_get_translation(t);

    auto w = spatial_top(x);
    auto v = spatial_bottom(x);

    v = quat_rotate(q, v);
    w = quat_rotate(q, w) + cross(p, v);

    return {w, v};
}

CUDA_CALLABLE vec3f vec_min(vec3f a, vec3f b) {
    return {min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2])};
}

CUDA_CALLABLE vec3f vec_max(vec3f a, vec3f b) {
    return {max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2])};
}

CUDA_CALLABLE vec3f vec_abs(vec3f a) {
    return {abs(a[0]), abs(a[1]), abs(a[2])};
}
}// namespace wp