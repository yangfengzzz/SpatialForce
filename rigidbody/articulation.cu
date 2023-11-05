//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "articulation.h"
#include "utils.h"

namespace wp {
__global__ void eval_articulation_fk(const array_t<int> &articulation_start,
                                     const array_t<int> &articulation_mask,
                                     const array_t<float> &joint_q,
                                     const array_t<float> &joint_qd,
                                     const array_t<int> &joint_q_start,
                                     const array_t<int> &joint_qd_start,
                                     const array_t<int> &joint_type,
                                     const array_t<int> &joint_parent,
                                     const array_t<int> &joint_child,
                                     const array_t<transformf> &joint_X_p,
                                     const array_t<transformf> &joint_X_c,
                                     const array_t<vec3f> &joint_axis,
                                     const array_t<int> &joint_axis_start,
                                     const array_t<int> &joint_axis_dim,
                                     const array_t<vec3f> &body_com,
                                     array_t<transformf> &body_q,
                                     array_t<spatial_vectorf> &body_qd) {
    int tid = wp::tid();

    // early out if disabling FK for this articulation
    if (articulation_mask(tid) == 0) {
        return;
    }

    auto joint_start = articulation_start(tid);
    auto joint_end = articulation_start(tid + 1);

    for (int i = joint_start; i < joint_end; i++) {
        auto parent = joint_parent[i];
        auto child = joint_child[i];
        auto X_wp = transform_identity();
        auto v_wp = spatial_vector();

        if (parent >= 0) {
            X_wp = body_q[parent];
            v_wp = body_qd[parent];
        }
        // compute transform across the joint
        auto type = joint_type[i];

        auto X_pj = joint_X_p[i];
        auto X_cj = joint_X_c[i];

        auto q_start = joint_q_start[i];
        auto qd_start = joint_qd_start[i];
        auto axis_start = joint_axis_start[i];
        auto lin_axis_count = joint_axis_dim(i, 0);
        auto ang_axis_count = joint_axis_dim(i, 1);

        auto X_jc = transform_identity();
        auto v_jc = spatial_vector(vec3(), vec3());

        if (type == int(JointType::JOINT_PRISMATIC)) {
            auto axis = joint_axis[axis_start];

            auto q = joint_q[q_start];
            auto qd = joint_qd[qd_start];

            X_jc = transform(axis * q, quat_identity());
            v_jc = spatial_vector(vec3(), axis * qd);
        }
        if (type == int(JointType::JOINT_REVOLUTE)) {
            auto axis = joint_axis[axis_start];

            auto q = joint_q[q_start];
            auto qd = joint_qd[qd_start];

            X_jc = transform(vec3(), quat_from_axis_angle(axis, q));
            v_jc = spatial_vector(axis * qd, vec3());
        }
        if (type == int(JointType::JOINT_BALL)) {
            auto r = quat(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2], joint_q[q_start + 3]);

            auto w = vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]);

            X_jc = transform(vec3(), r);
            v_jc = spatial_vector(w, vec3());
        }
        if (type == int(JointType::JOINT_FREE)) {
            auto t = transform(
                vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
                quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]));

            auto v = spatial_vector(
                vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),
                vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]));

            X_jc = t;
            v_jc = v;
        }
        if (type == int(JointType::JOINT_COMPOUND)) {
            auto axis = joint_axis[axis_start];

            // reconstruct rotation axes, todo: can probably use fact that rz'*ry'*rx' == rx*ry*rz to avoid some work here
            auto axis_0 = axis;
            auto q_0 = quat_from_axis_angle(axis_0, joint_q[q_start + 0]);

            auto axis_1 = joint_axis[axis_start + 1];
            auto q_1 = quat_from_axis_angle(axis_1, joint_q[q_start + 1]);

            auto axis_2 = joint_axis[axis_start + 2];
            auto q_2 = quat_from_axis_angle(axis_2, joint_q[q_start + 2]);

            auto t = transform(vec3(), q_2 * q_1 * q_0);

            auto v = spatial_vector(
                axis_0 * joint_qd[qd_start + 0] + axis_1 * joint_qd[qd_start + 1] + axis_2 * joint_qd[qd_start + 2],
                vec3());

            X_jc = t;
            v_jc = v;
        }
        if (type == int(JointType::JOINT_UNIVERSAL)) {
            // reconstruct rotation axes
            auto axis_0 = joint_axis[axis_start];
            auto q_0 = quat_from_axis_angle(axis_0, joint_q[q_start + 0]);

            auto axis_1 = joint_axis[axis_start + 1];
            auto q_1 = quat_from_axis_angle(axis_1, joint_q[q_start + 1]);

            auto t = transform(vec3(), q_1 * q_0);

            auto v = spatial_vector(axis_0 * joint_qd[qd_start + 0] + axis_1 * joint_qd[qd_start + 1], vec3());

            X_jc = t;
            v_jc = v;
        }
        if (type == int(JointType::JOINT_D6)) {
            auto pos = vec3(0.0);
            auto rot = quat_identity();
            auto vel_v = vec3(0.0);
            auto vel_w = vec3(0.0);

            // unroll for loop to ensure joint actions remain differentiable
            // (since differentiating through a for loop that updates a local variable is not supported)

            if (lin_axis_count > 0) {
                auto axis = joint_axis[axis_start + 0];
                pos += axis * joint_q[q_start + 0];
                vel_v += axis * joint_qd[qd_start + 0];
            }
            if (lin_axis_count > 1) {
                auto axis = joint_axis[axis_start + 1];
                pos += axis * joint_q[q_start + 1];
                vel_v += axis * joint_qd[qd_start + 1];
            }
            if (lin_axis_count > 2) {
                auto axis = joint_axis[axis_start + 2];
                pos += axis * joint_q[q_start + 2];
                vel_v += axis * joint_qd[qd_start + 2];
            }
            if (ang_axis_count > 0) {
                auto axis = joint_axis[axis_start + lin_axis_count + 0];
                auto qi = quat_from_axis_angle(axis, joint_q[q_start + lin_axis_count + 0]);
                rot = qi * rot;
                vel_w += joint_qd[qd_start + lin_axis_count + 0] * axis;
            }
            if (ang_axis_count > 1) {
                auto axis = joint_axis[axis_start + lin_axis_count + 1];
                auto qi = quat_from_axis_angle(axis, joint_q[q_start + lin_axis_count + 1]);
                rot = qi * rot;
                vel_w += joint_qd[qd_start + lin_axis_count + 1] * axis;
            }
            if (ang_axis_count > 2) {
                auto axis = joint_axis[axis_start + lin_axis_count + 2];
                auto qi = quat_from_axis_angle(axis, joint_q[q_start + lin_axis_count + 2]);
                rot = qi * rot;
                vel_w += joint_qd[qd_start + lin_axis_count + 2] * axis;
            }
            X_jc = transform(pos, rot);
            v_jc = spatial_vector(vel_w, vel_v);
        }
        auto X_wj = X_wp * X_pj;
        auto X_wc = X_wj * X_jc;
        X_wc *= transform_inverse(X_cj);

        // transform velocity across the joint to world space
        auto angular_vel = transform_vector(X_wj, spatial_top(v_jc));
        auto linear_vel = transform_vector(X_wj, spatial_bottom(v_jc));

        auto v_wc = v_wp + spatial_vector(angular_vel, linear_vel + cross(angular_vel, body_com[i]));

        body_q[child] = X_wc;
        body_qd[child] = v_wc;
    }
}

void eval_fk(Model &model, array_t<float> &joint_q, array_t<float> &joint_qd, array_t<float> &mask, State &state) {
}

__global__ void eval_articulation_ik(const array_t<transformf> &body_q,
                                     const array_t<spatial_vectorf> &body_qd,
                                     const array_t<vec3f> &body_com,
                                     const array_t<int> &joint_type,
                                     const array_t<int> &joint_parent,
                                     const array_t<int> &joint_child,
                                     const array_t<transformf> &joint_X_p,
                                     const array_t<transformf> &joint_X_c,
                                     const array_t<vec3f> &joint_axis,
                                     const array_t<int> &joint_axis_start,
                                     const array_t<int> &joint_axis_dim,
                                     const array_t<int> &joint_q_start,
                                     const array_t<int> &joint_qd_start,
                                     array_t<float> &joint_q,
                                     array_t<float> &joint_qd) {
    auto tid = wp::tid();

    auto c_parent = joint_parent[tid];
    auto c_child = joint_child[tid];

    auto X_pj = joint_X_p[tid];
    auto X_cj = joint_X_c[tid];

    auto X_wp = X_pj;
    auto r_p = vec3();
    auto w_p = vec3();
    auto v_p = vec3();

    // parent transform and moment arm
    if (c_parent >= 0) {
        X_wp = body_q[c_parent] * X_wp;
        auto r_wp = transform_get_translation(X_wp) - transform_point(body_q[c_parent], body_com[c_parent]);

        auto twist_p = body_qd[c_parent];

        w_p = spatial_top(twist_p);
        v_p = spatial_bottom(twist_p) + cross(w_p, r_wp);
    }
    // child transform and moment arm
    auto X_wc = body_q[c_child] * joint_X_c[tid];
    auto r_c = transform_get_translation(X_wc) - transform_point(body_q[c_child], body_com[c_child]);

    auto twist_c = body_qd[c_child];

    auto w_c = spatial_top(twist_c);
    auto v_c = spatial_bottom(twist_c) + cross(w_c, r_c);

    // joint properties
    auto type = joint_type[tid];

    auto x_p = transform_get_translation(X_wp);
    auto x_c = transform_get_translation(X_wc);

    auto q_p = transform_get_rotation(X_wp);
    auto q_c = transform_get_rotation(X_wc);

    // translational error
    auto x_err = x_c - x_p;
    auto v_err = v_c - v_p;
    auto w_err = w_c - w_p;

    auto q_start = joint_q_start[tid];
    auto qd_start = joint_qd_start[tid];
    auto axis_start = joint_axis_start[tid];
    auto lin_axis_count = joint_axis_dim(tid, 0);
    auto ang_axis_count = joint_axis_dim(tid, 1);

    if (type == int(JointType::JOINT_PRISMATIC)) {
        auto axis = joint_axis[axis_start];

        // world space joint axis
        auto axis_p = transform_vector(X_wp, axis);

        // evaluate joint coordinates
        auto q = dot(x_err, axis_p);
        auto qd = dot(v_err, axis_p);

        joint_q[q_start] = q;
        joint_qd[qd_start] = qd;

        return;
    }
    if (type == int(JointType::JOINT_REVOLUTE)) {
        auto axis = joint_axis[axis_start];

        auto axis_p = transform_vector(X_wp, axis);
        auto axis_c = transform_vector(X_wc, axis);

        // swing twist decomposition
        auto q_pc = quat_inverse(q_p) * q_c;
        auto twist = quat_twist(axis, q_pc);

        auto q = acos(twist.w) * 2.f * sign(dot(axis, vec3(twist.x, twist.y, twist.z)));
        auto qd = dot(w_err, axis_p);

        joint_q[q_start] = q;
        joint_qd[qd_start] = qd;

        return;
    }
    if (type == int(JointType::JOINT_BALL)) {
        auto q_pc = quat_inverse(q_p) * q_c;

        joint_q[q_start + 0] = q_pc[0];
        joint_q[q_start + 1] = q_pc[1];
        joint_q[q_start + 2] = q_pc[2];
        joint_q[q_start + 3] = q_pc[3];

        joint_qd[qd_start + 0] = w_err[0];
        joint_qd[qd_start + 1] = w_err[1];
        joint_qd[qd_start + 2] = w_err[2];

        return;
    }
    if (type == int(JointType::JOINT_FIXED)) {
        return;
    }
    if (type == int(JointType::JOINT_FREE)) {
        auto q_pc = quat_inverse(q_p) * q_c;

        joint_q[q_start + 0] = x_err[0];
        joint_q[q_start + 1] = x_err[1];
        joint_q[q_start + 2] = x_err[2];

        joint_q[q_start + 3] = q_pc[0];
        joint_q[q_start + 4] = q_pc[1];
        joint_q[q_start + 5] = q_pc[2];
        joint_q[q_start + 6] = q_pc[3];

        joint_qd[qd_start + 0] = w_err[0];
        joint_qd[qd_start + 1] = w_err[1];
        joint_qd[qd_start + 2] = w_err[2];

        joint_qd[qd_start + 3] = v_err[0];
        joint_qd[qd_start + 4] = v_err[1];
        joint_qd[qd_start + 5] = v_err[2];
    }
    if (type == int(JointType::JOINT_COMPOUND)) {
        auto q_off = transform_get_rotation(X_cj);
        auto q_pc = quat_inverse(q_off) * quat_inverse(q_p) * q_c * q_off;

        // decompose to a compound rotation each axis
        auto angles = quat_decompose(q_pc);

        // reconstruct rotation axes
        auto axis_0 = vec3(1.0, 0.0, 0.0);
        auto q_0 = quat_from_axis_angle(axis_0, angles[0]);

        auto axis_1 = quat_rotate(q_0, vec3(0.0, 1.0, 0.0));
        auto q_1 = quat_from_axis_angle(axis_1, angles[1]);

        auto axis_2 = quat_rotate(q_1 * q_0, vec3(0.0, 0.0, 1.0));
        auto q_2 = quat_from_axis_angle(axis_2, angles[2]);

        auto q_w = q_p * q_off;

        joint_q[q_start + 0] = angles[0];
        joint_q[q_start + 1] = angles[1];
        joint_q[q_start + 2] = angles[2];

        joint_qd[qd_start + 0] = dot(quat_rotate(q_w, axis_0), w_err);
        joint_qd[qd_start + 1] = dot(quat_rotate(q_w, axis_1), w_err);
        joint_qd[qd_start + 2] = dot(quat_rotate(q_w, axis_2), w_err);

        return;
    }
    if (type == int(JointType::JOINT_UNIVERSAL)) {
        auto q_off = transform_get_rotation(X_cj);
        auto q_pc = quat_inverse(q_off) * quat_inverse(q_p) * q_c * q_off;

        // decompose to a compound rotation each axis
        auto angles = quat_decompose(q_pc);

        // reconstruct rotation axes
        auto axis_0 = vec3(1.0, 0.0, 0.0);
        auto q_0 = quat_from_axis_angle(axis_0, angles[0]);

        auto axis_1 = quat_rotate(q_0, vec3(0.0, 1.0, 0.0));
        auto q_1 = quat_from_axis_angle(axis_1, angles[1]);

        auto q_w = q_p * q_off;

        joint_q[q_start + 0] = angles[0];
        joint_q[q_start + 1] = angles[1];

        joint_qd[qd_start + 0] = dot(quat_rotate(q_w, axis_0), w_err);
        joint_qd[qd_start + 1] = dot(quat_rotate(q_w, axis_1), w_err);

        return;
    }
    if (type == int(JointType::JOINT_D6)) {
        if (lin_axis_count > 0) {
            auto axis = transform_vector(X_wp, joint_axis[axis_start + 0]);
            joint_q[q_start + 0] = dot(x_err, axis);
            joint_qd[qd_start + 0] = dot(v_err, axis);
        }
        if (lin_axis_count > 1) {
            auto axis = transform_vector(X_wp, joint_axis[axis_start + 1]);
            joint_q[q_start + 1] = dot(x_err, axis);
            joint_qd[qd_start + 1] = dot(v_err, axis);
        }
        if (lin_axis_count > 2) {
            auto axis = transform_vector(X_wp, joint_axis[axis_start + 2]);
            joint_q[q_start + 2] = dot(x_err, axis);
            joint_qd[qd_start + 2] = dot(v_err, axis);
        }
        auto q_pc = quat_inverse(q_p) * q_c;
        if (ang_axis_count > 0) {
            auto axis = transform_vector(X_wp, joint_axis[axis_start + lin_axis_count + 0]);
            joint_q[q_start + lin_axis_count + 0] = quat_twist_angle(axis, q_pc);
            joint_qd[qd_start + lin_axis_count + 0] = dot(w_err, axis);
        }
        if (ang_axis_count > 1) {
            auto axis = transform_vector(X_wp, joint_axis[axis_start + lin_axis_count + 1]);
            joint_q[q_start + lin_axis_count + 1] = quat_twist_angle(axis, q_pc);
            joint_qd[qd_start + lin_axis_count + 1] = dot(w_err, axis);
        }
        if (ang_axis_count > 2) {
            auto axis = transform_vector(X_wp, joint_axis[axis_start + lin_axis_count + 2]);
            joint_q[q_start + lin_axis_count + 2] = quat_twist_angle(axis, q_pc);
            joint_qd[qd_start + lin_axis_count + 2] = dot(w_err, axis);
        }
        return;
    }
}

void eval_ik(Model &model, State &state, array_t<float> &joint_q, array_t<float> &joint_qd) {}

}// namespace wp