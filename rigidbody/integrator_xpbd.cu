//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "integrator_xpbd.h"
#include "core/spatial.h"
#include "core/hashgrid.h"
#include "utils.h"

namespace wp {
__global__ void solve_particle_ground_contacts(array_t<vec3f> particle_x,
                                               array_t<vec3f> particle_v,
                                               array_t<float> invmass,
                                               array_t<float> particle_radius,
                                               array_t<uint32_t> particle_flags,
                                               float ke,
                                               float kd,
                                               float kf,
                                               float mu,
                                               array_t<float> ground,
                                               float dt,
                                               float relaxation,
                                               array_t<vec3f> delta) {
    auto tid = wp::tid();
    if ((particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0) {
        return;
    }
    auto wi = invmass[tid];
    if (wi == 0.0) {
        return;
    }
    auto x = particle_x[tid];
    auto v = particle_v[tid];

    auto n = vec3(ground[0], ground[1], ground[2]);
    auto c = min(dot(n, x) + ground[3] - particle_radius[tid], 0.f);

    if (c > 0.0) {
        return;
    }
    // normal
    auto lambda_n = c;
    auto delta_n = n * lambda_n;

    // friction
    auto vn = dot(n, v);
    auto vt = v - n * vn;

    auto lambda_f = max(mu * lambda_n, 0.f - length(vt) * dt);
    auto delta_f = normalize(vt) * lambda_f;

    atomic_add(delta, tid, (delta_f - delta_n) / wi * relaxation);
}

__global__ void apply_soft_restitution_ground(array_t<vec3f> particle_x_new,
                                              array_t<vec3f> particle_v_new,
                                              array_t<vec3f> particle_x_old,
                                              array_t<vec3f> particle_v_old,
                                              array_t<float> invmass,
                                              array_t<float> particle_radius,
                                              array_t<uint32_t> particle_flags,
                                              float restitution,
                                              array_t<float> ground,
                                              float dt,
                                              float relaxation,
                                              array_t<vec3f> particle_v_out) {
    auto tid = wp::tid();
    if ((particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0) {
        return;
    }
    auto wi = invmass[tid];
    if (wi == 0.0) {
        return;
    }
    // x_new = particle_x_new[tid];
    auto v_new = particle_v_new[tid];
    auto x_old = particle_x_old[tid];
    auto v_old = particle_v_old[tid];

    auto n = vec3(ground[0], ground[1], ground[2]);
    auto c = dot(n, x_old) + ground[3] - particle_radius[tid];

    if (c > 0.0) {
        return;
    }
    auto rel_vel_old = dot(n, v_old);
    auto rel_vel_new = dot(n, v_new);
    auto dv = n * max(-rel_vel_new + max(-restitution * rel_vel_old, 0.f), 0.f);

    atomic_add(particle_v_out, tid, dv / wi * relaxation);
}

__global__ void solve_particle_shape_contacts(array_t<vec3f> particle_x,
                                              array_t<vec3f> particle_v,
                                              array_t<float> particle_invmass,
                                              array_t<float> particle_radius,
                                              array_t<uint32_t> particle_flags,
                                              array_t<transformf> body_q,
                                              array_t<spatial_vectorf> body_qd,
                                              array_t<vec3f> body_com,
                                              array_t<float> body_m_inv,
                                              array_t<mat33f> body_I_inv,
                                              array_t<int> shape_body,
                                              ModelShapeMaterials shape_materials,
                                              float particle_mu,
                                              float particle_ka,
                                              array_t<int> contact_count,
                                              array_t<int> contact_particle,
                                              array_t<int> contact_shape,
                                              array_t<vec3f> contact_body_pos,
                                              array_t<vec3f> contact_body_vel,
                                              array_t<vec3f> contact_normal,
                                              int contact_max,
                                              float dt,
                                              float relaxation,
                                              array_t<vec3f> delta,
                                              array_t<spatial_vectorf> body_delta) {
    auto tid = wp::tid();
    if ((particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0) {
        return;
    }
    auto count = min(contact_max, contact_count[0]);
    if (tid >= count) {
        return;
    }
    auto shape_index = contact_shape[tid];
    auto body_index = shape_body[shape_index];
    auto particle_index = contact_particle[tid];

    auto px = particle_x[particle_index];
    auto pv = particle_v[particle_index];

    auto X_wb = transform_identity();
    auto X_com = vec3();

    if (body_index >= 0) {
        X_wb = body_q[body_index];
        X_com = body_com[body_index];
    }
    // body position in world space
    auto bx = transform_point(X_wb, contact_body_pos[tid]);
    auto r = bx - transform_point(X_wb, X_com);

    auto n = contact_normal[tid];
    auto c = dot(n, px - bx) - particle_radius[tid];

    if (c > particle_ka) {
        return;
    }
    // take average material properties of shape and particle parameters
    auto mu = 0.5f * (particle_mu + shape_materials.mu[shape_index]);

    // body velocity
    auto body_v_s = spatial_vector();
    if (body_index >= 0) {
        body_v_s = body_qd[body_index];
    }
    auto body_w = spatial_top(body_v_s);
    auto body_v = spatial_bottom(body_v_s);

    // compute the body velocity at the particle position
    auto bv = body_v + cross(body_w, r) + transform_vector(X_wb, contact_body_vel[tid]);

    // relative velocity
    auto v = pv - bv;

    // normal
    auto lambda_n = c;
    auto delta_n = n * lambda_n;

    // friction
    auto vn = dot(n, v);
    auto vt = v - n * vn;

    // compute inverse masses
    auto w1 = particle_invmass[particle_index];
    auto w2 = 0.f;
    if (body_index >= 0) {
        auto angular = cross(r, n);
        auto q = transform_get_rotation(X_wb);
        auto rot_angular = quat_rotate_inv(q, angular);
        auto I_inv = body_I_inv[body_index];
        w2 = body_m_inv[body_index] + dot(rot_angular, I_inv * rot_angular);
    }
    auto denom = w1 + w2;
    if (denom == 0.0) {
        return;
    }
    auto lambda_f = max(mu * lambda_n, -length(vt) * dt);
    auto delta_f = normalize(vt) * lambda_f;
    auto delta_total = (delta_f - delta_n) / denom * relaxation;

    atomic_add(delta, particle_index, delta_total);

    if (body_index >= 0) {
        auto delta_t = cross(r, delta_total);
        atomic_sub(body_delta, body_index, spatial_vector(delta_t, delta_total));
    }
}

__global__ void solve_particle_particle_contacts(uint64_t grid,
                                                 array_t<vec3f> particle_x,
                                                 array_t<vec3f> particle_v,
                                                 array_t<float> particle_invmass,
                                                 array_t<float> particle_radius,
                                                 array_t<uint32_t> particle_flags,
                                                 float k_mu,
                                                 float k_cohesion,
                                                 float max_radius,
                                                 float dt,
                                                 float relaxation,
                                                 array_t<vec3f> deltas) {
    auto tid = wp::tid();

    // order threads by cell
    auto i = hash_grid_point_id(grid, tid);
    if (i == -1) {
        // hash grid has not been built yet
        return;
    }
    if ((particle_flags[i] & PARTICLE_FLAG_ACTIVE) == 0) {
        return;
    }
    auto x = particle_x[i];
    auto v = particle_v[i];
    auto radius = particle_radius[i];
    auto w1 = particle_invmass[i];

    // particle contact
    auto query = hash_grid_query(grid, x, radius + max_radius + k_cohesion);
    auto index = int(0);

    auto delta = vec3(0.0);

    while (hash_grid_query_next(query, index)) {
        if ((particle_flags[index] & PARTICLE_FLAG_ACTIVE) != 0 && index != i) {
            // compute distance to point
            auto n = x - particle_x[index];
            auto d = length(n);
            auto err = d - radius - particle_radius[index];

            // compute inverse masses
            auto w2 = particle_invmass[index];
            auto denom = w1 + w2;

            if (err <= k_cohesion && denom > 0.0) {
                n = n / d;
                auto vrel = v - particle_v[index];

                // normal
                auto lambda_n = err;
                auto delta_n = n * lambda_n;

                // friction
                auto vn = dot(n, vrel);
                auto vt = v - n * vn;

                auto lambda_f = max(k_mu * lambda_n, -length(vt) * dt);
                auto delta_f = normalize(vt) * lambda_f;
                delta += (delta_f - delta_n) / denom;
            }
        }
    }
    atomic_add(deltas, i, delta * relaxation);
}

__global__ void solve_springs(array_t<vec3f> x,
                              array_t<vec3f> v,
                              array_t<float> invmass,
                              array_t<int> spring_indices,
                              array_t<float> spring_rest_lengths,
                              array_t<float> spring_stiffness,
                              array_t<float> spring_damping,
                              float dt,
                              array_t<float> lambdas,
                              array_t<vec3f> delta) {
    auto tid = wp::tid();

    auto i = spring_indices[tid * 2 + 0];
    auto j = spring_indices[tid * 2 + 1];

    auto ke = spring_stiffness[tid];
    auto kd = spring_damping[tid];
    auto rest = spring_rest_lengths[tid];

    auto xi = x[i];
    auto xj = x[j];

    auto vi = v[i];
    auto vj = v[j];

    auto xij = xi - xj;
    auto vij = vi - vj;

    auto l = length(xij);

    if (l == 0.0) {
        return;
    }
    auto n = xij / l;

    auto c = l - rest;
    auto grad_c_xi = n;
    auto grad_c_xj = -1.f * n;

    auto wi = invmass[i];
    auto wj = invmass[j];

    auto denom = wi + wj;

    // Note strict inequality for damping -- 0 damping is ok
    if (denom <= 0.0 || ke <= 0.0 || kd < 0.0) {
        return;
    }
    auto alpha = 1.f / (ke * dt * dt);
    auto gamma = kd / (ke * dt);

    auto grad_c_dot_v = dt * dot(grad_c_xi, vij);// Note: dt because from the paper we want x_i - x^n, not v...
    auto dlambda = -1.f * (c + alpha * lambdas[tid] + gamma * grad_c_dot_v) / ((1.f + gamma) * denom + alpha);

    auto dxi = wi * dlambda * grad_c_xi;
    auto dxj = wj * dlambda * grad_c_xj;

    lambdas[tid] = lambdas[tid] + dlambda;

    atomic_add(delta, i, dxi);
    atomic_add(delta, j, dxj);
}

__global__ void bending_constraint(array_t<vec3f> x,
                                   array_t<vec3f> v,
                                   array_t<float> invmass,
                                   array_t<int> indices,
                                   array_t<float> rest,
                                   array_t<float> bending_properties,
                                   float dt,
                                   array_t<float> lambdas,
                                   array_t<vec3f> delta) {
    auto tid = wp::tid();
    auto eps = 1.0e-6f;

    auto ke = bending_properties(tid, 0);
    auto kd = bending_properties(tid, 1);

    auto i = indices(tid, 0);
    auto j = indices(tid, 1);
    auto k = indices(tid, 2);
    auto l = indices(tid, 3);

    if (i == -1 || j == -1 || k == -1 || l == -1) {
        return;
    }
    auto rest_angle = rest[tid];

    auto x1 = x[i];
    auto x2 = x[j];
    auto x3 = x[k];
    auto x4 = x[l];

    auto v1 = v[i];
    auto v2 = v[j];
    auto v3 = v[k];
    auto v4 = v[l];

    auto w1 = invmass[i];
    auto w2 = invmass[j];
    auto w3 = invmass[k];
    auto w4 = invmass[l];

    auto n1 = cross(x3 - x1, x4 - x1);// normal to face 1
    auto n2 = cross(x4 - x2, x3 - x2);// normal to face 2

    auto n1_length = length(n1);
    auto n2_length = length(n2);

    if (n1_length < eps || n2_length < eps) {
        return;
    }
    n1 /= n1_length;
    n2 /= n2_length;

    auto cos_theta = dot(n1, n2);

    auto e = x4 - x3;
    auto e_hat = normalize(e);
    auto e_length = length(e);

    auto derivative_flip = sign(dot(cross(n1, n2), e));
    derivative_flip *= -1.0;
    auto angle = acos(cos_theta);

    auto grad_x1 = n1 * e_length * derivative_flip;
    auto grad_x2 = n2 * e_length * derivative_flip;
    auto grad_x3 = (n1 * dot(x1 - x4, e_hat) + n2 * dot(x2 - x4, e_hat)) * derivative_flip;
    auto grad_x4 = (n1 * dot(x3 - x1, e_hat) + n2 * dot(x3 - x2, e_hat)) * derivative_flip;
    auto c = angle - rest_angle;
    auto denominator = (w1 * length_sq(grad_x1) + w2 * length_sq(grad_x2) + w3 * length_sq(grad_x3) + w4 * length_sq(grad_x4));

    // Note strict inequality for damping -- 0 damping is ok
    if (denominator <= 0.0 || ke <= 0.0 || kd < 0.0) {
        return;
    }
    auto alpha = 1.f / (ke * dt * dt);
    auto gamma = kd / (ke * dt);

    auto grad_dot_v = dt * (dot(grad_x1, v1) + dot(grad_x2, v2) + dot(grad_x3, v3) + dot(grad_x4, v4));

    auto dlambda = -1.f * (c + alpha * lambdas[tid] + gamma * grad_dot_v) / ((1.f + gamma) * denominator + alpha);

    auto delta0 = w1 * dlambda * grad_x1;
    auto delta1 = w2 * dlambda * grad_x2;
    auto delta2 = w3 * dlambda * grad_x3;
    auto delta3 = w4 * dlambda * grad_x4;

    lambdas[tid] = lambdas[tid] + dlambda;

    atomic_add(delta, i, delta0);
    atomic_add(delta, j, delta1);
    atomic_add(delta, k, delta2);
    atomic_add(delta, l, delta3);
}

__global__ void solve_tetrahedra(array_t<vec3f> x,
                                 array_t<vec3f> v,
                                 array_t<float> inv_mass,
                                 array_t<int> indices,
                                 array_t<mat33f> pose,
                                 array_t<float> activation,
                                 array_t<float> materials,
                                 float dt,
                                 float relaxation,
                                 array_t<vec3f> delta) {
    auto tid = wp::tid();

    auto i = indices(tid, 0);
    auto j = indices(tid, 1);
    auto k = indices(tid, 2);
    auto l = indices(tid, 3);

    auto act = activation[tid];

    auto k_mu = materials(tid, 0);
    auto k_lambda = materials(tid, 1);
    auto k_damp = materials(tid, 2);

    auto x0 = x[i];
    auto x1 = x[j];
    auto x2 = x[k];
    auto x3 = x[l];

    auto v0 = v[i];
    auto v1 = v[j];
    auto v2 = v[k];
    auto v3 = v[l];

    auto w0 = inv_mass[i];
    auto w1 = inv_mass[j];
    auto w2 = inv_mass[k];
    auto w3 = inv_mass[l];

    auto x10 = x1 - x0;
    auto x20 = x2 - x0;
    auto x30 = x3 - x0;

    auto v10 = v1 - v0;
    auto v20 = v2 - v0;
    auto v30 = v3 - v0;

    auto Ds = mat33(x10, x20, x30);
    auto Dm = pose[tid];

    auto inv_rest_volume = determinant(Dm) * 6.f;
    auto rest_volume = 1.f / inv_rest_volume;

    // F = Xs*Xm^-1
    auto F = Ds * Dm;

    auto f1 = vec3(F(0, 0), F(1, 0), F(2, 0));
    auto f2 = vec3(F(0, 1), F(1, 1), F(2, 1));
    auto f3 = vec3(F(0, 2), F(1, 2), F(2, 2));

    // C_sqrt
    // tr = dot(f1, f1) + dot(f2, f2) + dot(f3, f3);
    // r_s = sqrt(abs(tr - 3.0));
    // C = r_s;

    // if (r_s == 0.0):
    // return;

    // if (tr < 3.0):
    // r_s = 0.0 - r_s;

    // dCdx = F*transpose(Dm)*(1.0/r_s);
    // alpha = 1.0 + k_mu / k_lambda;

    // C_Neo
    auto r_s = sqrt(dot(f1, f1) + dot(f2, f2) + dot(f3, f3));
    if (r_s == 0.0) {
        return;
    }
    // tr = dot(f1, f1) + dot(f2, f2) + dot(f3, f3);
    // if (tr < 3.0):
    // r_s = -r_s;
    auto r_s_inv = 1.f / r_s;
    auto C = r_s;
    auto dCdx = F * transpose(Dm) * r_s_inv;
    auto alpha = 1.f + k_mu / k_lambda;

    // C_Spherical
    // r_s = sqrt(dot(f1, f1) + dot(f2, f2) + dot(f3, f3));
    // r_s_inv = 1.0/r_s;
    // C = r_s - sqrt(3.0);
    // dCdx = F*transpose(Dm)*r_s_inv;
    // alpha = 1.0;

    // C_D
    // r_s = sqrt(dot(f1, f1) + dot(f2, f2) + dot(f3, f3));
    // C = r_s*r_s - 3.0;
    // dCdx = F*transpose(Dm)*2.0;
    // alpha = 1.0;

    auto grad1 = vec3(dCdx(0, 0), dCdx(1, 0), dCdx(2, 0));
    auto grad2 = vec3(dCdx(0, 1), dCdx(1, 1), dCdx(2, 1));
    auto grad3 = vec3(dCdx(0, 2), dCdx(1, 2), dCdx(2, 2));
    auto grad0 = (grad1 + grad2 + grad3) * (0.f - 1.f);

    auto denom = (dot(grad0, grad0) * w0 + dot(grad1, grad1) * w1 + dot(grad2, grad2) * w2 + dot(grad3, grad3) * w3);
    auto multiplier = C / (denom + 1.f / (k_mu * dt * dt * rest_volume));

    auto delta0 = grad0 * multiplier;
    auto delta1 = grad1 * multiplier;
    auto delta2 = grad2 * multiplier;
    auto delta3 = grad3 * multiplier;

    // hydrostatic part
    auto J = determinant(F);

    auto C_vol = J - alpha;
    // dCdx = mat33(cross(f2, f3), cross(f3, f1), cross(f1, f2))*transpose(Dm);

    // grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0]);
    // grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1]);
    // grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2]);
    // grad0 = (grad1 + grad2 + grad3)*(0.0 - 1.0);

    auto s = inv_rest_volume / 6.f;
    grad1 = cross(x20, x30) * s;
    grad2 = cross(x30, x10) * s;
    grad3 = cross(x10, x20) * s;
    grad0 = -(grad1 + grad2 + grad3);

    denom = (dot(grad0, grad0) * w0 + dot(grad1, grad1) * w1 + dot(grad2, grad2) * w2 + dot(grad3, grad3) * w3);
    multiplier = C_vol / (denom + 1.f / (k_lambda * dt * dt * rest_volume));

    delta0 += grad0 * multiplier;
    delta1 += grad1 * multiplier;
    delta2 += grad2 * multiplier;
    delta3 += grad3 * multiplier;

    // apply forces
    atomic_sub(delta, i, delta0 * w0 * relaxation);
    atomic_sub(delta, j, delta1 * w1 * relaxation);
    atomic_sub(delta, k, delta2 * w2 * relaxation);
    atomic_sub(delta, l, delta3 * w3 * relaxation);
}

__global__ void apply_particle_deltas(array_t<vec3f> x_orig,
                                      array_t<vec3f> x_pred,
                                      array_t<uint32_t> particle_flags,
                                      array_t<vec3f> delta,
                                      float dt,
                                      float v_max,
                                      array_t<vec3f> x_out,
                                      array_t<vec3f> v_out) {
    auto tid = wp::tid();
    if ((particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0) {
        return;
    }
    auto x0 = x_orig[tid];
    auto xp = x_pred[tid];

    // constraint deltas
    auto d = delta[tid];

    auto x_new = xp + d;
    auto v_new = (x_new - x0) / dt;

    // enforce velocity limit to prevent instability
    auto v_new_mag = length(v_new);
    if (v_new_mag > v_max) {
        v_new *= v_max / v_new_mag;
    }
    x_out[tid] = x_new;
    v_out[tid] = v_new;
}

__global__ void apply_body_deltas(array_t<transformf> q_in,
                                  array_t<spatial_vectorf> qd_in,
                                  array_t<vec3f> body_com,
                                  array_t<mat33> body_I,
                                  array_t<float> body_inv_m,
                                  array_t<mat33> body_inv_I,
                                  array_t<spatial_vectorf> deltas,
                                  array_t<float> constraint_inv_weights,
                                  float dt,
                                  array_t<transformf> q_out,
                                  array_t<spatial_vectorf> qd_out) {
    auto tid = wp::tid();
    auto inv_m = body_inv_m[tid];
    if (inv_m == 0.0) {
        q_out[tid] = q_in[tid];
        qd_out[tid] = qd_in[tid];
        return;
    }
    auto inv_I = body_inv_I[tid];

    auto tf = q_in[tid];
    auto delta = deltas[tid];

    auto p0 = transform_get_translation(tf);
    auto q0 = transform_get_rotation(tf);

    auto weight = 1.f;
    if (constraint_inv_weights)
        if (constraint_inv_weights[tid] > 0.0)
            weight = 1.f / constraint_inv_weights[tid];

    auto dp = spatial_bottom(delta) * (inv_m * weight);
    auto dq = spatial_top(delta) * weight;
    dq = quat_rotate(q0, inv_I * quat_rotate_inv(q0, dq));

    // update orientation
    auto q1 = q0 + 0.5f * quat(dq * dt * dt, 0.0) * q0;
    q1 = normalize(q1);

    // update position
    auto com = body_com[tid];
    auto x_com = p0 + quat_rotate(q0, com);
    auto p1 = x_com + dp * dt * dt;
    p1 -= quat_rotate(q1, com);

    q_out[tid] = transform(p1, q1);

    auto v0 = spatial_bottom(qd_in[tid]);
    auto w0 = spatial_top(qd_in[tid]);

    // update linear and angular velocity
    auto v1 = v0 + dp * dt;
    // angular part (compute in body frame)
    auto wb = quat_rotate_inv(q0, w0 + dq * dt);
    auto tb = -cross(wb, body_I[tid] * wb);// coriolis forces
    auto w1 = quat_rotate(q0, wb + inv_I * tb * dt);

    qd_out[tid] = spatial_vector(w1, v1);
}

__global__ void apply_body_delta_velocities(array_t<spatial_vectorf> qd_in,
                                            array_t<spatial_vectorf> deltas,
                                            array_t<spatial_vectorf> qd_out) {
    auto tid = wp::tid();
    qd_out[tid] = qd_in[tid] + deltas[tid];
}

__global__ void apply_joint_torques(array_t<transformf> body_q,
                                    array_t<vec3f> body_com,
                                    array_t<int> joint_q_start,
                                    array_t<int> joint_qd_start,
                                    array_t<int> joint_type,
                                    array_t<int> joint_parent,
                                    array_t<int> joint_child,
                                    array_t<transformf> joint_X_p,
                                    array_t<transformf> joint_X_c,
                                    array_t<int> joint_axis_start,
                                    array_t<int> joint_axis_dim,
                                    array_t<vec3f> joint_axis,
                                    array_t<float> joint_act,
                                    array_t<spatial_vectorf> body_f) {
    auto tid = wp::tid();
    auto type = joint_type[tid];
    if (type == int(JointType::JOINT_FIXED)) {
        return;
    }
    if (type == int(JointType::JOINT_FREE)) {
        return;
    }
    if (type == int(JointType::JOINT_DISTANCE)) {
        return;
    }
    if (type == int(JointType::JOINT_BALL)) {
        return;
    }

    // rigid body indices of the child and parent
    auto id_c = joint_child[tid];
    auto id_p = joint_parent[tid];

    auto X_pj = joint_X_p[tid];
    auto X_cj = joint_X_c[tid];

    auto X_wp = X_pj;
    auto pose_p = X_pj;
    auto com_p = vec3(0.0);
    // parent transform and moment arm
    if (id_p >= 0) {
        pose_p = body_q[id_p];
        X_wp = pose_p * X_wp;
        com_p = body_com[id_p];
    }
    auto r_p = transform_get_translation(X_wp) - transform_point(pose_p, com_p);

    // child transform and moment arm
    auto pose_c = body_q[id_c];
    auto X_wc = pose_c;
    auto com_c = body_com[id_c];
    auto r_c = transform_get_translation(X_wc) - transform_point(pose_c, com_c);

    // local joint rotations
    auto q_p = transform_get_rotation(X_wp);
    auto q_c = transform_get_rotation(X_wc);

    // joint properties (for 1D joints)
    auto q_start = joint_q_start[tid];
    auto qd_start = joint_qd_start[tid];
    auto axis_start = joint_axis_start[tid];
    auto lin_axis_count = joint_axis_dim(tid, 0);
    auto ang_axis_count = joint_axis_dim(tid, 1);

    // total force/torque on the parent
    auto t_total = vec3();
    auto f_total = vec3();

    if (type == int(JointType::JOINT_REVOLUTE)) {
        auto axis = joint_axis[axis_start];
        auto act = joint_act[qd_start];
        auto a_p = transform_vector(X_wp, axis);
        t_total += act * a_p;
    } else if (type == int(JointType::JOINT_PRISMATIC)) {
        auto axis = joint_axis[axis_start];
        auto act = joint_act[qd_start];
        auto a_p = transform_vector(X_wp, axis);
        f_total += act * a_p;
    } else if (type == int(JointType::JOINT_COMPOUND)) {
        auto axis_0 = joint_axis[axis_start + 0];
        auto axis_1 = joint_axis[axis_start + 1];
        auto axis_2 = joint_axis[axis_start + 2];
        t_total += joint_act[qd_start + 0] * transform_vector(X_wp, axis_0);
        t_total += joint_act[qd_start + 1] * transform_vector(X_wp, axis_1);
        t_total += joint_act[qd_start + 2] * transform_vector(X_wp, axis_2);
    } else if (type == int(JointType::JOINT_UNIVERSAL)) {
        auto axis_0 = joint_axis[axis_start + 0];
        auto axis_1 = joint_axis[axis_start + 1];
        t_total += joint_act[qd_start + 0] * transform_vector(X_wp, axis_0);
        t_total += joint_act[qd_start + 1] * transform_vector(X_wp, axis_1);
    } else if (type == int(JointType::JOINT_D6)) {
        if (lin_axis_count > 0) {
            auto axis = joint_axis[axis_start + 0];
            auto act = joint_act[qd_start + 0];
            auto a_p = transform_vector(X_wp, axis);
            f_total += act * a_p;
        }
        if (lin_axis_count > 1) {
            auto axis = joint_axis[axis_start + 1];
            auto act = joint_act[qd_start + 1];
            auto a_p = transform_vector(X_wp, axis);
            f_total += act * a_p;
        }
        if (lin_axis_count > 2) {
            auto axis = joint_axis[axis_start + 2];
            auto act = joint_act[qd_start + 2];
            auto a_p = transform_vector(X_wp, axis);
            f_total += act * a_p;
        }
        if (ang_axis_count > 0) {
            auto axis = joint_axis[axis_start + lin_axis_count + 0];
            auto act = joint_act[qd_start + lin_axis_count + 0];
            auto a_p = transform_vector(X_wp, axis);
            t_total += act * a_p;
        }
        if (ang_axis_count > 1) {
            auto axis = joint_axis[axis_start + lin_axis_count + 1];
            auto act = joint_act[qd_start + lin_axis_count + 1];
            auto a_p = transform_vector(X_wp, axis);
            t_total += act * a_p;
        }
        if (ang_axis_count > 2) {
            auto axis = joint_axis[axis_start + lin_axis_count + 2];
            auto act = joint_act[qd_start + lin_axis_count + 2];
            auto a_p = transform_vector(X_wp, axis);
            t_total += act * a_p;
        }
    } else {
        print("joint type not handled in apply_joint_torques");
    }

    // write forces
    if (id_p >= 0) {
        atomic_add(body_f, id_p, spatial_vector(t_total + cross(r_p, f_total), f_total));
    }
    atomic_sub(body_f, id_c, spatial_vector(t_total + cross(r_c, f_total), f_total));
}

CUDA_CALLABLE_DEVICE vec3ub update_joint_axis_mode(uint8_t mode, vec3f axis, vec_t<3, uint8_t> input_axis_mode) {
    // update the 3D axis mode flags given the axis vector and mode of this axis
    auto mode_x = max(uint8_t(uint8_t(nonzero(axis[0])) * mode), input_axis_mode[0]);
    auto mode_y = max(uint8_t(uint8_t(nonzero(axis[1])) * mode), input_axis_mode[1]);
    auto mode_z = max(uint8_t(uint8_t(nonzero(axis[2])) * mode), input_axis_mode[2]);
    return {mode_x, mode_y, mode_z};
}

CUDA_CALLABLE_DEVICE spatial_vectorf update_joint_axis_limits(vec3f axis, float limit_lower, float limit_upper, spatial_vectorf input_limits) {
    // update the 3D linear/angular limits (spatial_vector [lower, upper]) given the axis vector and limits
    auto lo_temp = axis * limit_lower;
    auto up_temp = axis * limit_upper;
    auto lo = vec_min(lo_temp, up_temp);
    auto up = vec_max(lo_temp, up_temp);
    auto input_lower = spatial_top(input_limits);
    auto input_upper = spatial_bottom(input_limits);
    auto lower = vec_min(input_lower, lo);
    auto upper = vec_max(input_upper, up);
    return {lower, upper};
}

CUDA_CALLABLE_DEVICE mat33 update_joint_axis_target_ke_kd(vec3f axis, float target, float target_ke, float target_kd, mat33f input_target_ke_kd) {
    // update the 3D linear/angular target, target_ke, and target_kd (mat33 [target, ke, kd]) given the axis vector and target, target_ke, target_kd
    auto axis_target = input_target_ke_kd[0];
    auto axis_ke = input_target_ke_kd[1];
    auto axis_kd = input_target_ke_kd[2];
    auto stiffness = axis * target_ke;
    axis_target += stiffness * target;// weighted target (to be normalized later by sum of target_ke)
    axis_ke += vec_abs(stiffness);
    axis_kd += vec_abs(axis * target_kd);
    return {
        axis_target[0],
        axis_target[1],
        axis_target[2],
        axis_ke[0],
        axis_ke[1],
        axis_ke[2],
        axis_kd[0],
        axis_kd[1],
        axis_kd[2]};
}

CUDA_CALLABLE_DEVICE float compute_contact_constraint_delta(float err, transformf tf_a, transformf tf_b, float m_inv_a, float m_inv_b,
                                                            mat33f I_inv_a, mat33f I_inv_b, vec3f linear_a, vec3f linear_b,
                                                            vec3f angular_a, vec3f angular_b, float relaxation, float dt) {
    auto denom = 0.f;
    denom += length_sq(linear_a) * m_inv_a;
    denom += length_sq(linear_b) * m_inv_b;

    auto q1 = transform_get_rotation(tf_a);
    auto q2 = transform_get_rotation(tf_b);

    // Eq. 2-3 (make sure to project into the frame of the body)
    auto rot_angular_a = quat_rotate_inv(q1, angular_a);
    auto rot_angular_b = quat_rotate_inv(q2, angular_b);

    denom += dot(rot_angular_a, I_inv_a * rot_angular_a);
    denom += dot(rot_angular_b, I_inv_b * rot_angular_b);

    auto deltaLambda = -err;
    if (denom > 0.0) {
        deltaLambda /= dt * dt * denom;
    }
    return deltaLambda * relaxation;
}

CUDA_CALLABLE_DEVICE float compute_positional_correction(float err, float derr, transformf tf_a, transformf tf_b,
                                                         float m_inv_a, float m_inv_b, mat33f I_inv_a, mat33f I_inv_b,
                                                         vec3f linear_a, vec3f linear_b, vec3f angular_a, vec3f angular_b,
                                                         float lambda_in, float compliance, float damping, float dt) {
    auto denom = 0.f;
    denom += length_sq(linear_a) * m_inv_a;
    denom += length_sq(linear_b) * m_inv_b;

    auto q1 = transform_get_rotation(tf_a);
    auto q2 = transform_get_rotation(tf_b);

    // Eq. 2-3 (make sure to project into the frame of the body)
    auto rot_angular_a = quat_rotate_inv(q1, angular_a);
    auto rot_angular_b = quat_rotate_inv(q2, angular_b);

    denom += dot(rot_angular_a, I_inv_a * rot_angular_a);
    denom += dot(rot_angular_b, I_inv_b * rot_angular_b);

    auto alpha = compliance;
    auto gamma = compliance * damping;

    auto deltaLambda = -(err + alpha * lambda_in + gamma * derr);
    if (denom + alpha > 0.0) {
        deltaLambda /= dt * (dt + gamma) * denom + alpha;
    }
    return deltaLambda;
}

CUDA_CALLABLE_DEVICE float compute_angular_correction(float err, float derr, transformf tf_a, transformf tf_b,
                                                      mat33f I_inv_a, mat33f I_inv_b, vec3f angular_a, vec3f angular_b,
                                                      float lambda_in, float compliance, float damping, float dt) {
    auto denom = 0.f;

    auto q1 = transform_get_rotation(tf_a);
    auto q2 = transform_get_rotation(tf_b);

    // Eq. 2-3 (make sure to project into the frame of the body)
    auto rot_angular_a = quat_rotate_inv(q1, angular_a);
    auto rot_angular_b = quat_rotate_inv(q2, angular_b);

    denom += dot(rot_angular_a, I_inv_a * rot_angular_a);
    denom += dot(rot_angular_b, I_inv_b * rot_angular_b);

    auto alpha = compliance;
    auto gamma = compliance * damping;

    auto deltaLambda = -(err + alpha * lambda_in + gamma * derr);
    if (denom + alpha > 0.0) {
        deltaLambda /= dt * (dt + gamma) * denom + alpha;
    }
    return deltaLambda;
}

__global__ void solve_body_joints(array_t<transformf> body_q,
                                  array_t<spatial_vectorf> body_qd,
                                  array_t<vec3f> body_com,
                                  array_t<float> body_inv_m,
                                  array_t<mat33f> body_inv_I,
                                  array_t<int> joint_type,
                                  array_t<int> joint_enabled,
                                  array_t<int> joint_parent,
                                  array_t<int> joint_child,
                                  array_t<transformf> joint_X_p,
                                  array_t<transformf> joint_X_c,
                                  array_t<float> joint_limit_lower,
                                  array_t<float> joint_limit_upper,
                                  array_t<int> joint_axis_start,
                                  array_t<int> joint_axis_dim,
                                  array_t<int8_t> joint_axis_mode,
                                  array_t<vec3f> joint_axis,
                                  array_t<float> joint_target,
                                  array_t<float> joint_target_ke,
                                  array_t<float> joint_target_kd,
                                  array_t<float> joint_linear_compliance,
                                  array_t<float> joint_angular_compliance,
                                  float angular_relaxation,
                                  float linear_relaxation,
                                  float dt,
                                  array_t<spatial_vectorf> deltas) {
    auto tid = wp::tid();
    auto type = joint_type[tid];

    if (joint_enabled[tid] == 0 || type == int(JointType::JOINT_FREE)) {
        return;
    }
    // rigid body indices of the child and parent
    auto id_c = joint_child[tid];
    auto id_p = joint_parent[tid];

    auto X_pj = joint_X_p[tid];
    auto X_cj = joint_X_c[tid];

    auto X_wp = X_pj;
    auto m_inv_p = 0.f;
    auto I_inv_p = mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    auto pose_p = X_pj;
    auto com_p = vec3(0.0);
    auto vel_p = vec3(0.0);
    auto omega_p = vec3(0.0);
    // parent transform and moment arm
    if (id_p >= 0) {
        pose_p = body_q[id_p];
        X_wp = pose_p * X_wp;
        com_p = body_com[id_p];
        m_inv_p = body_inv_m[id_p];
        I_inv_p = body_inv_I[id_p];
        vel_p = spatial_bottom(body_qd[id_p]);
        omega_p = spatial_top(body_qd[id_p]);
    }
    // child transform and moment arm
    auto pose_c = body_q[id_c];
    auto X_wc = pose_c * X_cj;
    auto com_c = body_com[id_c];
    auto m_inv_c = body_inv_m[id_c];
    auto I_inv_c = body_inv_I[id_c];
    auto vel_c = spatial_bottom(body_qd[id_c]);
    auto omega_c = spatial_top(body_qd[id_c]);

    if (m_inv_p == 0.0 && m_inv_c == 0.0) {
        // connection between two immovable bodies
        return;
    }
    // accumulate constraint deltas
    auto lin_delta_p = vec3(0.0);
    auto ang_delta_p = vec3(0.0);
    auto lin_delta_c = vec3(0.0);
    auto ang_delta_c = vec3(0.0);

    auto rel_pose = transform_inverse(X_wp) * X_wc;
    auto rel_p = transform_get_translation(rel_pose);

    // joint connection points
    // x_p = transform_get_translation(X_wp);
    auto x_c = transform_get_translation(X_wc);

    auto linear_compliance = joint_linear_compliance[tid];
    auto angular_compliance = joint_angular_compliance[tid];

    auto axis_start = joint_axis_start[tid];
    auto lin_axis_count = joint_axis_dim(tid, 0);
    auto ang_axis_count = joint_axis_dim(tid, 1);

    auto world_com_p = transform_point(pose_p, com_p);
    auto world_com_c = transform_point(pose_c, com_c);

    // handle positional constraints
    if (type == int(JointType::JOINT_DISTANCE)) {
        auto r_p = transform_get_translation(X_wp) - world_com_p;
        auto r_c = transform_get_translation(X_wc) - world_com_c;
        auto lower = joint_limit_lower[axis_start];
        auto upper = joint_limit_upper[axis_start];
        if (lower < 0.0 && upper < 0.0) {
            // no limits
            return;
        }
        auto d = length(rel_p);
        auto err = 0.f;
        if (lower >= 0.0 && d < lower) {
            err = d - lower;
            // use a more descriptive direction vector for the constraint
            // in case the joint parent and child anchors are very close
            rel_p = err * normalize(world_com_c - world_com_p);
        } else if (upper >= 0.0 && d > upper) {
            err = d - upper;
        }

        if (abs(err) > 1e-9) {
            // compute gradients
            auto linear_c = rel_p;
            auto linear_p = -linear_c;
            r_c = x_c - world_com_c;
            auto angular_p = -cross(r_p, linear_c);
            auto angular_c = cross(r_c, linear_c);
            // constraint time derivative
            auto derr = (dot(linear_p, vel_p) + dot(linear_c, vel_c) + dot(angular_p, omega_p) + dot(angular_c, omega_c));
            auto lambda_in = 0.f;
            auto compliance = linear_compliance;
            auto ke = joint_target_ke[axis_start];
            if (ke > 0.0)
                compliance = 1.f / ke;
            auto damping = joint_target_kd[axis_start];
            auto d_lambda = compute_positional_correction(
                err,
                derr,
                pose_p,
                pose_c,
                m_inv_p,
                m_inv_c,
                I_inv_p,
                I_inv_c,
                linear_p,
                linear_c,
                angular_p,
                angular_c,
                lambda_in,
                compliance,
                damping,
                dt);

            lin_delta_p += linear_p * (d_lambda * linear_relaxation);
            ang_delta_p += angular_p * (d_lambda * angular_relaxation);
            lin_delta_c += linear_c * (d_lambda * linear_relaxation);
            ang_delta_c += angular_c * (d_lambda * angular_relaxation);
        }
    } else {
        // compute joint target, stiffness, damping
        auto ke_sum = float(0.0);
        auto axis_limits = spatial_vector({0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
        auto axis_mode = vec3ub(uint8(0), uint8(0), uint8(0));
        auto axis_target_ke_kd = mat33(0.0);
        // avoid a for loop here since local variables would need to be modified which is not yet differentiable
        if (lin_axis_count > 0) {
            auto axis = joint_axis[axis_start];
            auto lo_temp = axis * joint_limit_lower[axis_start];
            auto up_temp = axis * joint_limit_upper[axis_start];
            axis_limits = spatial_vector(vec_min(lo_temp, up_temp), vec_max(lo_temp, up_temp));
            auto mode = joint_axis_mode[axis_start];
            if (mode != int(JointMode::JOINT_MODE_LIMIT)) {
                //  position or velocity target
                auto ke = joint_target_ke[axis_start];
                auto kd = joint_target_kd[axis_start];
                auto target = joint_target[axis_start];
                axis_mode = update_joint_axis_mode(mode, axis, axis_mode);
                axis_target_ke_kd = update_joint_axis_target_ke_kd(axis, target, ke, kd, axis_target_ke_kd);
                ke_sum += ke;
            }
        }
        if (lin_axis_count > 1) {
            auto axis_idx = axis_start + 1;
            auto axis = joint_axis[axis_idx];
            auto lower = joint_limit_lower[axis_idx];
            auto upper = joint_limit_upper[axis_idx];
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits);
            auto mode = joint_axis_mode[axis_idx];
            if (mode != int(JointMode::JOINT_MODE_LIMIT)) {
                // position or velocity target
                auto ke = joint_target_ke[axis_idx];
                auto kd = joint_target_kd[axis_idx];
                auto target = joint_target[axis_idx];
                axis_mode = update_joint_axis_mode(mode, axis, axis_mode);
                axis_target_ke_kd = update_joint_axis_target_ke_kd(axis, target, ke, kd, axis_target_ke_kd);
                ke_sum += ke;
            }
        }
        if (lin_axis_count > 2) {
            auto axis_idx = axis_start + 2;
            auto axis = joint_axis[axis_idx];
            auto lower = joint_limit_lower[axis_idx];
            auto upper = joint_limit_upper[axis_idx];
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits);
            auto mode = joint_axis_mode[axis_idx];
            if (mode != int(JointMode::JOINT_MODE_LIMIT)) {
                // position or velocity target
                auto ke = joint_target_ke[axis_idx];
                auto kd = joint_target_kd[axis_idx];
                auto target = joint_target[axis_idx];
                axis_mode = update_joint_axis_mode(mode, axis, axis_mode);
                axis_target_ke_kd = update_joint_axis_target_ke_kd(axis, target, ke, kd, axis_target_ke_kd);
                ke_sum += ke;
            }
        }
        auto axis_target = axis_target_ke_kd[0];
        auto axis_stiffness = axis_target_ke_kd[1];
        auto axis_damping = axis_target_ke_kd[2];
        if (ke_sum > 0.0) {
            axis_target /= ke_sum;
        }
        auto axis_limits_lower = spatial_top(axis_limits);
        auto axis_limits_upper = spatial_bottom(axis_limits);

        auto frame_p = quat_to_matrix(transform_get_rotation(X_wp));
        // note that x_c appearing in both is correct
        auto r_p = x_c - world_com_p;
        auto r_c = x_c - transform_point(pose_c, com_c);

        // for loop will be unrolled, so we can modify local variables
        for (int dim = 0; dim < 3; dim++) {
            auto e = rel_p[dim];
            auto mode = axis_mode[dim];

            // compute gradients
            auto linear_c = vec3(frame_p(0, dim), frame_p(1, dim), frame_p(2, dim));
            auto linear_p = -linear_c;
            auto angular_p = -cross(r_p, linear_c);
            auto angular_c = cross(r_c, linear_c);
            // constraint time derivative
            auto derr = (dot(linear_p, vel_p) + dot(linear_c, vel_c) + dot(angular_p, omega_p) + dot(angular_c, omega_c));

            auto err = 0.f;
            auto compliance = linear_compliance;
            auto damping = 0.f;
            // consider joint limits irrespective of axis mode
            auto lower = axis_limits_lower[dim];
            auto upper = axis_limits_upper[dim];
            if (e < lower) {
                err = e - lower;
                compliance = linear_compliance;
                damping = 0.0;
            } else if (e > upper) {
                err = e - upper;
                compliance = linear_compliance;
                damping = 0.0;
            } else {
                auto target = axis_target[dim];
                if (mode == int(JointMode::JOINT_MODE_TARGET_POSITION)) {
                    target = clamp(target, lower, upper);
                    if (axis_stiffness[dim] > 0.0) {
                        err = e - target;
                        compliance = 1.f / axis_stiffness[dim];
                    }
                    damping = axis_damping[dim];
                } else if (mode == int(JointMode::JOINT_MODE_TARGET_VELOCITY)) {
                    if (axis_stiffness[dim] > 0.0) {
                        err = (derr - target) * dt;
                        compliance = 1.f / axis_stiffness[dim];
                    }
                    damping = axis_damping[dim];
                }
            }
            if (abs(err) > 1e-9) {
                auto lambda_in = 0.f;
                auto d_lambda = compute_positional_correction(
                    err,
                    derr,
                    pose_p,
                    pose_c,
                    m_inv_p,
                    m_inv_c,
                    I_inv_p,
                    I_inv_c,
                    linear_p,
                    linear_c,
                    angular_p,
                    angular_c,
                    lambda_in,
                    compliance,
                    damping,
                    dt);

                lin_delta_p += linear_p * (d_lambda * linear_relaxation);
                ang_delta_p += angular_p * (d_lambda * angular_relaxation);
                lin_delta_c += linear_c * (d_lambda * linear_relaxation);
                ang_delta_c += angular_c * (d_lambda * angular_relaxation);
            }
        }
    }

    if (type == int(JointType::JOINT_FIXED) || type == int(JointType::JOINT_PRISMATIC) || type == int(JointType::JOINT_REVOLUTE) || type == int(JointType::JOINT_UNIVERSAL) || type == int(JointType::JOINT_COMPOUND) || type == int(JointType::JOINT_D6)) {
        // handle angular constraints

        // local joint rotations
        auto q_p = transform_get_rotation(X_wp);
        auto q_c = transform_get_rotation(X_wc);

        // make quats lie in same hemisphere
        if (dot(q_p, q_c) < 0.0) {
            q_c *= -1.f;
        }
        auto rel_q = quat_inverse(q_p) * q_c;

        auto qtwist = normalize(quat(rel_q[0], 0.0, 0.0, rel_q[3]));
        auto qswing = rel_q * quat_inverse(qtwist);

        // decompose to a compound rotation each axis
        auto s = sqrt(rel_q[0] * rel_q[0] + rel_q[3] * rel_q[3]);
        auto invs = 1.f / s;
        auto invscube = invs * invs * invs;

        // handle axis-angle joints

        // rescale twist from quaternion space to angular
        auto err_0 = 2.f * asin(clamp(qtwist[0], -1.f, 1.f));
        auto err_1 = qswing[1];
        auto err_2 = qswing[2];
        // analytic gradients of swing-twist decomposition
        auto grad_0 = quat(invs - rel_q[0] * rel_q[0] * invscube, 0.0, 0.0, -(rel_q[3] * rel_q[0]) * invscube);
        auto grad_1 = quat(
            -rel_q[3] * (rel_q[3] * rel_q[2] + rel_q[0] * rel_q[1]) * invscube,
            rel_q[3] * invs,
            -rel_q[0] * invs,
            rel_q[0] * (rel_q[3] * rel_q[2] + rel_q[0] * rel_q[1]) * invscube);
        auto grad_2 = quat(
            rel_q[3] * (rel_q[3] * rel_q[1] - rel_q[0] * rel_q[2]) * invscube,
            rel_q[0] * invs,
            rel_q[3] * invs,
            rel_q[0] * (rel_q[2] * rel_q[0] - rel_q[3] * rel_q[1]) * invscube);
        grad_0 *= 2.f / abs(qtwist[3]);
        // grad_0 *= 2.0 / sqrt(1.0-qtwist[0]*qtwist[0]);	// derivative of asin(x) = 1/sqrt(1-x^2)

        // rescale swing
        auto swing_sq = qswing[3] * qswing[3];
        // if swing axis magnitude close to zero vector, just treat in quaternion space
        auto angularEps = 1.0e-4;
        if (swing_sq + angularEps < 1.0) {
            auto d = sqrt(1.f - qswing[3] * qswing[3]);
            auto theta = 2.f * acos(clamp(qswing[3], -1.f, 1.f));
            auto scale = theta / d;

            err_1 *= scale;
            err_2 *= scale;

            grad_1 *= scale;
            grad_2 *= scale;
        }
        auto errs = vec3(err_0, err_1, err_2);
        auto grad_x = vec3(grad_0[0], grad_1[0], grad_2[0]);
        auto grad_y = vec3(grad_0[1], grad_1[1], grad_2[1]);
        auto grad_z = vec3(grad_0[2], grad_1[2], grad_2[2]);
        auto grad_w = vec3(grad_0[3], grad_1[3], grad_2[3]);

        // compute joint target, stiffness, damping
        auto ke_sum = float(0.0);
        auto axis_limits = spatial_vector({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        auto axis_mode = vec3ub(uint8(0), uint8(0), uint8(0));
        auto axis_target_ke_kd = mat33(0.0);
        // avoid a for loop here since local variables would need to be modified which is not yet differentiable
        if (ang_axis_count > 0) {
            auto axis_idx = axis_start + lin_axis_count;
            auto axis = joint_axis[axis_idx];
            auto lo_temp = axis * joint_limit_lower[axis_idx];
            auto up_temp = axis * joint_limit_upper[axis_idx];
            axis_limits = spatial_vector(vec_min(lo_temp, up_temp), vec_max(lo_temp, up_temp));
            auto mode = joint_axis_mode[axis_idx];
            if (mode != int(JointMode::JOINT_MODE_LIMIT)) {// position or velocity target
                auto ke = joint_target_ke[axis_idx];
                auto kd = joint_target_kd[axis_idx];
                auto target = joint_target[axis_idx];
                axis_mode = update_joint_axis_mode(mode, axis, axis_mode);
                axis_target_ke_kd = update_joint_axis_target_ke_kd(axis, target, ke, kd, axis_target_ke_kd);
                ke_sum += ke;
            }
        }
        if (ang_axis_count > 1) {
            auto axis_idx = axis_start + lin_axis_count + 1;
            auto axis = joint_axis[axis_idx];
            auto lower = joint_limit_lower[axis_idx];
            auto upper = joint_limit_upper[axis_idx];
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits);
            auto mode = joint_axis_mode[axis_idx];
            if (mode != int(JointMode::JOINT_MODE_LIMIT)) {// position or velocity target
                auto ke = joint_target_ke[axis_idx];
                auto kd = joint_target_kd[axis_idx];
                auto target = joint_target[axis_idx];
                axis_mode = update_joint_axis_mode(mode, axis, axis_mode);
                axis_target_ke_kd = update_joint_axis_target_ke_kd(axis, target, ke, kd, axis_target_ke_kd);
                ke_sum += ke;
            }
        }
        if (ang_axis_count > 2) {
            auto axis_idx = axis_start + lin_axis_count + 2;
            auto axis = joint_axis[axis_idx];
            auto lower = joint_limit_lower[axis_idx];
            auto upper = joint_limit_upper[axis_idx];
            axis_limits = update_joint_axis_limits(axis, lower, upper, axis_limits);
            auto mode = joint_axis_mode[axis_idx];
            if (mode != int(JointMode::JOINT_MODE_LIMIT)) {// position or velocity target
                auto ke = joint_target_ke[axis_idx];
                auto kd = joint_target_kd[axis_idx];
                auto target = joint_target[axis_idx];
                axis_mode = update_joint_axis_mode(mode, axis, axis_mode);
                axis_target_ke_kd = update_joint_axis_target_ke_kd(axis, target, ke, kd, axis_target_ke_kd);
                ke_sum += ke;
            }
        }

        auto axis_target = axis_target_ke_kd[0];
        auto axis_stiffness = axis_target_ke_kd[1];
        auto axis_damping = axis_target_ke_kd[2];
        if (ke_sum > 0.0) {
            axis_target /= ke_sum;
        }
        auto axis_limits_lower = spatial_top(axis_limits);
        auto axis_limits_upper = spatial_bottom(axis_limits);

        for (int dim = 0; dim < 3; dim++) {
            auto e = errs[dim];
            auto mode = axis_mode[dim];

            // analytic gradients of swing-twist decomposition
            auto grad = quat(grad_x[dim], grad_y[dim], grad_z[dim], grad_w[dim]);

            auto quat_c = 0.5f * q_p * grad * quat_inverse(q_c);
            auto angular_c = vec3(quat_c[0], quat_c[1], quat_c[2]);
            auto angular_p = -angular_c;
            // time derivative of the constraint
            auto derr = dot(angular_p, omega_p) + dot(angular_c, omega_c);

            auto err = 0.f;
            auto compliance = angular_compliance;
            auto damping = 0.f;

            // consider joint limits irrespective of mode
            auto lower = axis_limits_lower[dim];
            auto upper = axis_limits_upper[dim];
            if (e < lower) {
                err = e - lower;
                compliance = angular_compliance;
                damping = 0.0;
            } else if (e > upper) {
                err = e - upper;
                compliance = angular_compliance;
                damping = 0.0;
            } else {
                auto target = axis_target[dim];
                if (mode == int(JointMode::JOINT_MODE_TARGET_POSITION)) {
                    target = clamp(target, lower, upper);
                    if (axis_stiffness[dim] > 0.0) {
                        err = e - target;
                        compliance = 1.f / axis_stiffness[dim];
                    }
                    damping = axis_damping[dim];
                } else if (mode == int(JointMode::JOINT_MODE_TARGET_VELOCITY)) {
                    if (axis_stiffness[dim] > 0.0) {
                        err = (derr - target) * dt;
                        compliance = 1.f / axis_stiffness[dim];
                    }
                    damping = axis_damping[dim];
                }
            }
            auto d_lambda = (compute_angular_correction(err, derr, pose_p, pose_c, I_inv_p, I_inv_c,
                                                        angular_p, angular_c,
                                                        0.0, compliance, damping, dt) *
                             angular_relaxation);
            // update deltas
            ang_delta_p += angular_p * d_lambda;
            ang_delta_c += angular_c * d_lambda;
        }
    }

    if (id_p >= 0)
        atomic_add(deltas, id_p, spatial_vector(ang_delta_p, lin_delta_p));
    if (id_c >= 0)
        atomic_add(deltas, id_c, spatial_vector(ang_delta_c, lin_delta_c));
}

__global__ void solve_body_contact_positions(array_t<transformf> body_q,
                                             array_t<spatial_vectorf> body_qd,
                                             array_t<vec3f> body_com,
                                             array_t<float> body_m_inv,
                                             array_t<mat33f> body_I_inv,
                                             array_t<int> contact_count,
                                             array_t<int> contact_body0,
                                             array_t<int> contact_body1,
                                             array_t<vec3f> contact_point0,
                                             array_t<vec3f> contact_point1,
                                             array_t<vec3f> contact_offset0,
                                             array_t<vec3f> contact_offset1,
                                             array_t<vec3f> contact_normal,
                                             array_t<float> contact_thickness,
                                             array_t<int> contact_shape0,
                                             array_t<int> contact_shape1,
                                             ModelShapeMaterials shape_materials,
                                             float relaxation,
                                             float dt,
                                             float contact_torsional_friction,
                                             float contact_rolling_friction,
                                             array_t<spatial_vectorf> deltas,
                                             array_t<vec3f> active_contact_point0,
                                             array_t<vec3f> active_contact_point1,
                                             array_t<float> active_contact_distance,
                                             array_t<float> contact_inv_weight) {
    auto tid = wp::tid();

    auto count = contact_count[0];
    if (tid >= count) {
        return;
    }
    auto body_a = contact_body0[tid];
    auto body_b = contact_body1[tid];

    if (body_a == body_b) {
        return;
    }
    if (contact_shape0[tid] == contact_shape1[tid]) {
        return;
    }

    // find body to world transform
    auto X_wb_a = transform_identity();
    auto X_wb_b = transform_identity();
    if (body_a >= 0) {
        X_wb_a = body_q[body_a];
    }
    if (body_b >= 0) {
        X_wb_b = body_q[body_b];
    }
    // compute body position in world space
    auto bx_a = transform_point(X_wb_a, contact_point0[tid]);
    auto bx_b = transform_point(X_wb_b, contact_point1[tid]);
    active_contact_point0[tid] = bx_a;
    active_contact_point1[tid] = bx_b;

    auto thickness = contact_thickness[tid];
    auto n = -contact_normal[tid];
    auto d = dot(n, bx_b - bx_a) - thickness;

    active_contact_distance[tid] = d;

    if (d >= 0.0) {
        return;
    }
    auto m_inv_a = 0.f;
    auto m_inv_b = 0.f;
    auto I_inv_a = mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    auto I_inv_b = mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    // center of mass in body frame
    auto com_a = vec3(0.0);
    auto com_b = vec3(0.0);
    // body to world transform
    X_wb_a = transform_identity();
    X_wb_b = transform_identity();
    // angular velocities
    auto omega_a = vec3(0.0);
    auto omega_b = vec3(0.0);
    // contact offset in body frame
    auto offset_a = contact_offset0[tid];
    auto offset_b = contact_offset1[tid];

    if (body_a >= 0) {
        X_wb_a = body_q[body_a];
        com_a = body_com[body_a];
        m_inv_a = body_m_inv[body_a];
        I_inv_a = body_I_inv[body_a];
        omega_a = spatial_top(body_qd[body_a]);
    }
    if (body_b >= 0) {
        X_wb_b = body_q[body_b];
        com_b = body_com[body_b];
        m_inv_b = body_m_inv[body_b];
        I_inv_b = body_I_inv[body_b];
        omega_b = spatial_top(body_qd[body_b]);
    }
    // use average contact material properties
    auto mat_nonzero = 0;
    auto mu = 0.f;
    auto shape_a = contact_shape0[tid];
    auto shape_b = contact_shape1[tid];
    if (shape_a >= 0) {
        mat_nonzero += 1;
        mu += shape_materials.mu[shape_a];
    }
    if (shape_b >= 0) {
        mat_nonzero += 1;
        mu += shape_materials.mu[shape_b];
    }
    if (mat_nonzero > 0) {
        mu /= float(mat_nonzero);
    }
    auto r_a = bx_a - transform_point(X_wb_a, com_a);
    auto r_b = bx_b - transform_point(X_wb_b, com_b);

    auto angular_a = -cross(r_a, n);
    auto angular_b = cross(r_b, n);

    if (contact_inv_weight) {
        if (body_a >= 0) {
            atomic_add(contact_inv_weight, body_a, 1.f);
        }
        if (body_b >= 0) {
            atomic_add(contact_inv_weight, body_b, 1.f);
        }
    }
    auto lambda_n = compute_contact_constraint_delta(
        d, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b,
        -n, n, angular_a, angular_b, relaxation, dt);

    auto lin_delta_a = -n * lambda_n;
    auto lin_delta_b = n * lambda_n;
    auto ang_delta_a = angular_a * lambda_n;
    auto ang_delta_b = angular_b * lambda_n;

    // linear friction
    if (mu > 0.0) {
        // add on displacement from surface offsets, this ensures we include any rotational effects due to thickness from feature
        // need to use the current rotation to account for friction due to angular effects (e.g.: slipping contact)
        bx_a += transform_vector(X_wb_a, offset_a);
        bx_b += transform_vector(X_wb_b, offset_b);

        // update delta
        auto delta = bx_b - bx_a;
        auto friction_delta = delta - dot(n, delta) * n;

        auto perp = normalize(friction_delta);

        r_a = bx_a - transform_point(X_wb_a, com_a);
        r_b = bx_b - transform_point(X_wb_b, com_b);

        angular_a = -cross(r_a, perp);
        angular_b = cross(r_b, perp);

        auto err = length(friction_delta);

        if (err > 0.0) {
            auto lambda_fr = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, -perp, perp, angular_a, angular_b, 1.0, dt);

            // limit friction based on incremental normal force, good approximation to limiting on total force
            lambda_fr = max(lambda_fr, -lambda_n * mu);

            lin_delta_a -= perp * lambda_fr;
            lin_delta_b += perp * lambda_fr;

            ang_delta_a += angular_a * lambda_fr;
            ang_delta_b += angular_b * lambda_fr;
        }
    }

    auto torsional_friction = mu * contact_torsional_friction;

    auto delta_omega = omega_b - omega_a;

    if (torsional_friction > 0.0) {
        auto err = dot(delta_omega, n) * dt;

        if (abs(err) > 0.0) {
            auto lin = vec3(0.0);
            auto lambda_torsion = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b, lin, lin, -n, n, 1.0, dt);

            lambda_torsion = clamp(lambda_torsion, -lambda_n * torsional_friction, lambda_n * torsional_friction);

            ang_delta_a -= n * lambda_torsion;
            ang_delta_b += n * lambda_torsion;
        }
    }

    auto rolling_friction = mu * contact_rolling_friction;
    if (rolling_friction > 0.0) {
        delta_omega -= dot(n, delta_omega) * n;
        auto err = length(delta_omega) * dt;
        if (err > 0.0) {
            auto lin = vec3(0.0);
            auto roll_n = normalize(delta_omega);
            auto lambda_roll = compute_contact_constraint_delta(
                err, X_wb_a, X_wb_b, m_inv_a, m_inv_b, I_inv_a, I_inv_b,
                lin, lin, -roll_n, roll_n, 1.0, dt);

            lambda_roll = max(lambda_roll, -lambda_n * rolling_friction);

            ang_delta_a -= roll_n * lambda_roll;
            ang_delta_b += roll_n * lambda_roll;
        }
    }
    if (body_a >= 0) {
        atomic_add(deltas, body_a, spatial_vector(ang_delta_a, lin_delta_a));
    }
    if (body_b >= 0) {
        atomic_add(deltas, body_b, spatial_vector(ang_delta_b, lin_delta_b));
    }
}

__global__ void update_body_velocities(array_t<transformf> poses,
                                       array_t<transformf> poses_prev,
                                       array_t<vec3f> body_com,
                                       float dt,
                                       array_t<spatial_vectorf> qd_out) {
    auto tid = wp::tid();

    auto pose = poses[tid];
    auto pose_prev = poses_prev[tid];

    auto x = transform_get_translation(pose);
    auto x_prev = transform_get_translation(pose_prev);

    auto q = transform_get_rotation(pose);
    auto q_prev = transform_get_rotation(pose_prev);

    // Update body velocities according to Alg. 2
    // XXX we consider the body COM as the origin of the body frame
    auto x_com = x + quat_rotate(q, body_com[tid]);
    auto x_com_prev = x_prev + quat_rotate(q_prev, body_com[tid]);

    // XXX consider the velocity of the COM
    auto v = (x_com - x_com_prev) / dt;
    auto dq = q * quat_inverse(q_prev);

    auto omega = 2.f / dt * vec3(dq[0], dq[1], dq[2]);
    if (dq[3] < 0.f) {
        omega = -omega;
    }
    qd_out[tid] = spatial_vector(omega, v);
}

__global__ void apply_rigid_restitution(array_t<transformf> body_q,
                                        array_t<spatial_vectorf> body_qd,
                                        array_t<transformf> body_q_prev,
                                        array_t<spatial_vectorf> body_qd_prev,
                                        array_t<vec3f> body_com,
                                        array_t<float> body_m_inv,
                                        array_t<mat33f> body_I_inv,
                                        array_t<int> contact_count,
                                        array_t<int> contact_body0,
                                        array_t<int> contact_body1,
                                        array_t<vec3f> contact_normal,
                                        array_t<int> contact_shape0,
                                        array_t<int> contact_shape1,
                                        ModelShapeMaterials shape_materials,
                                        array_t<float> active_contact_distance,
                                        array_t<vec3f> active_contact_point0,
                                        array_t<vec3f> active_contact_point1,
                                        array_t<float> contact_inv_weight,
                                        vec3f gravity,
                                        float dt,
                                        array_t<spatial_vectorf> deltas) {
    auto tid = wp::tid();

    auto count = contact_count[0];
    if (tid >= count) {
        return;
    }
    auto d = active_contact_distance[tid];
    if (d >= 0.0) {
        return;
    }
    // use average contact material properties
    auto mat_nonzero = 0;
    auto restitution = 0.f;
    auto shape_a = contact_shape0[tid];
    auto shape_b = contact_shape1[tid];
    if (shape_a >= 0) {
        mat_nonzero += 1;
        restitution += shape_materials.restitution[shape_a];
    }
    if (shape_b >= 0) {
        mat_nonzero += 1;
        restitution += shape_materials.restitution[shape_b];
    }
    if (mat_nonzero > 0) {
        restitution /= float(mat_nonzero);
    }
    auto body_a = contact_body0[tid];
    auto body_b = contact_body1[tid];

    auto m_inv_a = 0.f;
    auto m_inv_b = 0.f;
    auto I_inv_a = mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    auto I_inv_b = mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    // body to world transform
    auto X_wb_a_prev = transform_identity();
    auto X_wb_b_prev = transform_identity();
    // center of mass in body frame
    auto com_a = vec3(0.0);
    auto com_b = vec3(0.0);
    // previous velocity at contact points
    auto v_a = vec3(0.0);
    auto v_b = vec3(0.0);
    // new velocity at contact points
    auto v_a_new = vec3(0.0);
    auto v_b_new = vec3(0.0);
    // inverse mass used to compute the impulse
    auto inv_mass = 0.f;

    transformf X_wb_a, X_wb_b;
    if (body_a >= 0) {
        X_wb_a_prev = body_q_prev[body_a];
        X_wb_a = body_q[body_a];
        m_inv_a = body_m_inv[body_a];
        I_inv_a = body_I_inv[body_a];
        com_a = body_com[body_a];
    }
    if (body_b >= 0) {
        X_wb_b_prev = body_q_prev[body_b];
        X_wb_b = body_q[body_b];
        m_inv_b = body_m_inv[body_b];
        I_inv_b = body_I_inv[body_b];
        com_b = body_com[body_b];
    }
    auto bx_a = active_contact_point0[tid];
    auto bx_b = active_contact_point1[tid];

    auto r_a = bx_a - transform_point(X_wb_a, com_a);
    auto r_b = bx_b - transform_point(X_wb_b, com_b);

    auto n = contact_normal[tid];
    if (body_a >= 0) {
        v_a = velocity_at_point(body_qd_prev[body_a], r_a) + gravity * dt;
        v_a_new = velocity_at_point(body_qd[body_a], r_a);
        auto q_a = transform_get_rotation(X_wb_a_prev);
        auto rxn = quat_rotate_inv(q_a, cross(r_a, n));
        // Eq. 2
        auto inv_mass_a = m_inv_a + dot(rxn, I_inv_a * rxn);
        // if (contact_inv_weight):
        if (contact_inv_weight[body_a] > 0.0)
            inv_mass_a *= contact_inv_weight[body_a];
        inv_mass += inv_mass_a;
        // inv_mass += m_inv_a + dot(rxn, I_inv_a * rxn);
    }
    if (body_b >= 0) {
        v_b = velocity_at_point(body_qd_prev[body_b], r_b) + gravity * dt;
        v_b_new = velocity_at_point(body_qd[body_b], r_b);
        auto q_b = transform_get_rotation(X_wb_b_prev);
        auto rxn = quat_rotate_inv(q_b, cross(r_b, n));
        // Eq. 3
        auto inv_mass_b = m_inv_b + dot(rxn, I_inv_b * rxn);
        // if (contact_inv_weight):
        if (contact_inv_weight[body_b] > 0.0)
            inv_mass_b *= contact_inv_weight[body_b];
        inv_mass += inv_mass_b;
        // inv_mass += m_inv_b + dot(rxn, I_inv_b * rxn);
    }
    if (inv_mass == 0.0) {
        return;
    }
    // Eq. 29
    auto rel_vel_old = dot(n, v_a - v_b);
    auto rel_vel_new = dot(n, v_a_new - v_b_new);

    // Eq. 34 (Eq. 33 from the ACM paper, note the max operation)
    auto dv = n * (-rel_vel_new + max(-restitution * rel_vel_old, 0.f));

    // Eq. 33
    auto p = dv / inv_mass;
    if (body_a >= 0) {
        auto p_a = p;
        if (contact_inv_weight)
            if (contact_inv_weight[body_a] > 0.0)
                p_a /= contact_inv_weight[body_a];
        auto q_a = transform_get_rotation(X_wb_a);
        auto rxp = quat_rotate_inv(q_a, cross(r_a, p_a));
        auto dq = quat_rotate(q_a, I_inv_a * rxp);
        atomic_add(deltas, body_a, spatial_vector(dq, p_a * m_inv_a));
    }
    if (body_b >= 0) {
        auto p_b = p;
        if (contact_inv_weight)
            if (contact_inv_weight[body_b] > 0.0)
                p_b /= contact_inv_weight[body_b];
        auto q_b = transform_get_rotation(X_wb_b);
        auto rxp = quat_rotate_inv(q_b, cross(r_b, p_b));
        auto dq = quat_rotate(q_b, I_inv_b * rxp);
        atomic_sub(deltas, body_b, spatial_vector(dq, p_b * m_inv_b));
    }
}

XPBDIntegrator::XPBDIntegrator(int iterations,
                               float soft_body_relaxation,
                               float soft_contact_relaxation,
                               float joint_linear_relaxation,
                               float joint_angular_relaxation,
                               float rigid_contact_relaxation,
                               bool rigid_contact_con_weighting,
                               float angular_damping,
                               bool enable_restitution)
    : iterations_{iterations},
      soft_body_relaxation_{soft_body_relaxation},
      soft_contact_relaxation_{soft_contact_relaxation},
      joint_linear_relaxation_{joint_linear_relaxation},
      joint_angular_relaxation_{joint_angular_relaxation},
      rigid_contact_relaxation_{rigid_contact_relaxation},
      rigid_contact_con_weighting_{rigid_contact_con_weighting},
      angular_damping_{angular_damping},
      enable_restitution_{enable_restitution} {
}

void XPBDIntegrator::simulate(Model &model, State &state_in, State &state_out, float dt) {
}

}// namespace wp
