//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "collide.h"
#include "integrator_euler.h"
#include "core/spatial.h"
#include "utils.h"

namespace wp {
__global__ void integrate_particles(const array_t<vec3f> &x, const array_t<vec3f> &v, const array_t<vec3f> &f,
                                    const array_t<float> &w, const array_t<uint32_t> &particle_flags, vec3f gravity,
                                    float dt, float v_max, array_t<vec3f> &x_new, array_t<vec3f> &v_new) {
    auto tid = wp::tid();
    if ((particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0) {
        return;
    }
    auto x0 = x[tid];
    auto v0 = v[tid];
    auto f0 = f[tid];

    auto inv_mass = w[tid];

    // simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    auto v1 = v0 + (f0 * inv_mass + gravity * step(0.f - inv_mass)) * dt;
    // enforce velocity limit to prevent instability
    auto v1_mag = length(v1);
    if (v1_mag > v_max) {
        v1 *= v_max / v1_mag;
    }
    auto x1 = x0 + v1 * dt;

    x_new[tid] = x1;
    v_new[tid] = v1;
}

__global__ void integrate_bodies(array_t<transformf> body_q,
                                 array_t<spatial_vectorf> body_qd,
                                 array_t<spatial_vectorf> body_f,
                                 array_t<vec3f> body_com,
                                 array_t<float> m,
                                 array_t<mat33f> I,
                                 array_t<float> inv_m,
                                 array_t<mat33f> inv_I,
                                 vec3f gravity,
                                 float angular_damping,
                                 float dt,
                                 array_t<transformf> body_q_new,
                                 array_t<spatial_vectorf> body_qd_new) {
    auto tid = wp::tid();

    // positions
    auto q = body_q[tid];
    auto qd = body_qd[tid];
    auto f = body_f[tid];

    // masses
    auto mass = m[tid];
    auto inv_mass = inv_m[tid];// 1 / mass;

    auto inertia = I[tid];
    auto inv_inertia = inv_I[tid];// inverse of 3x3 inertia matrix

    // unpack transform
    auto x0 = transform_get_translation(q);
    auto r0 = transform_get_rotation(q);

    // unpack spatial twist
    auto w0 = spatial_top(qd);
    auto v0 = spatial_bottom(qd);

    // unpack spatial wrench
    auto t0 = spatial_top(f);
    auto f0 = spatial_bottom(f);

    auto x_com = x0 + quat_rotate(r0, body_com[tid]);

    // linear part
    auto v1 = v0 + (f0 * inv_mass + gravity * nonzero(inv_mass)) * dt;
    auto x1 = x_com + v1 * dt;

    // angular part (compute in body frame)
    auto wb = quat_rotate_inv(r0, w0);
    auto tb = quat_rotate_inv(r0, t0) - cross(wb, inertia * wb);// coriolis forces

    auto w1 = quat_rotate(r0, wb + inv_inertia * tb * dt);
    auto r1 = normalize(r0 + quat(w1, 0.0) * r0 * 0.5f * dt);

    // angular damping
    w1 *= 1.f - angular_damping * dt;

    body_q_new[tid] = transform(x1 - quat_rotate(r1, body_com[tid]), r1);
    body_qd_new[tid] = spatial_vector(w1, v1);
}

__global__ void eval_springs(array_t<vec3f> x,
                             array_t<vec3f> v,
                             array_t<int> spring_indices,
                             array_t<float> spring_rest_lengths,
                             array_t<float> spring_stiffness,
                             array_t<float> spring_damping,
                             array_t<vec3f> f) {
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
    auto l_inv = 1.f / l;

    // normalized spring direction
    auto dir = xij * l_inv;

    auto c = l - rest;
    auto dcdt = dot(dir, vij);

    // damping based on relative velocity.
    auto fs = dir * (ke * c + kd * dcdt);

    atomic_sub(f, i, fs);
    atomic_add(f, j, fs);
}

__global__ void eval_triangles(array_t<vec3f> x,
                               array_t<vec3f> v,
                               array_t<int> indices,
                               array_t<mat22f> pose,
                               array_t<float> activation,
                               array_t<float> materials,
                               array_t<vec3f> f) {
    auto tid = wp::tid();

    auto k_mu = materials(tid, 0);
    auto k_lambda = materials(tid, 1);
    auto k_damp = materials(tid, 2);
    auto k_drag = materials(tid, 3);
    auto k_lift = materials(tid, 4);

    auto i = indices(tid, 0);
    auto j = indices(tid, 1);
    auto k = indices(tid, 2);

    auto x0 = x[i];// point zero
    auto x1 = x[j];// point one
    auto x2 = x[k];// point two

    auto v0 = v[i];// vel zero
    auto v1 = v[j];// vel one
    auto v2 = v[k];// vel two

    auto x10 = x1 - x0;// barycentric coordinates (centered at p)
    auto x20 = x2 - x0;

    auto v10 = v1 - v0;
    auto v20 = v2 - v0;

    auto Dm = pose[tid];

    auto inv_rest_area = determinant(Dm) * 2.f;// 1 / det(A) = det(A^-1)
    auto rest_area = 1.f / inv_rest_area;

    // scale stiffness coefficients to account for area
    k_mu = k_mu * rest_area;
    k_lambda = k_lambda * rest_area;
    k_damp = k_damp * rest_area;

    // F = Xs*Xm^-1
    auto F1 = x10 * Dm(0, 0) + x20 * Dm(1, 0);
    auto F2 = x10 * Dm(0, 1) + x20 * Dm(1, 1);

    // dFdt = Vs*Xm^-1
    auto dFdt1 = v10 * Dm(0, 0) + v20 * Dm(1, 0);
    auto dFdt2 = v10 * Dm(0, 1) + v20 * Dm(1, 1);

    // deviatoric PK1 + damping term
    auto P1 = F1 * k_mu + dFdt1 * k_damp;
    auto P2 = F2 * k_mu + dFdt2 * k_damp;

    // -----------------------------
    // St. Venant-Kirchoff
    //
    // // Green strain, F'*F-I
    // e00 = dot(f1, f1) - 1.0;
    // e10 = dot(f2, f1);
    // e01 = dot(f1, f2);
    // e11 = dot(f2, f2) - 1.0;
    //
    // E = mat22(e00, e01,
    //  e10, e11);
    //
    // // local forces (deviatoric part)
    // T = mul(E, transpose(Dm));
    //
    // // spatial forces, F*T
    // fq = (f1*T[0,0] + f2*T[1,0])*k_mu*2.0;
    // fr = (f1*T[0,1] + f2*T[1,1])*k_mu*2.0;
    // alpha = 1.0;
    //
    // -----------------------------
    // Baraff & Witkin, note this model is not isotropic
    //
    // c1 = length(f1) - 1.0;
    // c2 = length(f2) - 1.0;
    // f1 = normalize(f1)*c1*k1;
    // f2 = normalize(f2)*c2*k1;
    //
    // fq = f1*Dm[0,0] + f2*Dm[0,1];
    // fr = f1*Dm[1,0] + f2*Dm[1,1];
    //
    // -----------------------------
    // Neo-Hookean (with rest stability)

    // force = P*Dm'
    auto f1 = P1 * Dm(0, 0) + P2 * Dm(0, 1);
    auto f2 = P1 * Dm(1, 0) + P2 * Dm(1, 1);
    auto alpha = 1.f + k_mu / k_lambda;

    // -----------------------------
    // Area Preservation

    auto n = cross(x10, x20);
    auto area = length(n) * 0.5f;

    // actuation
    auto act = activation[tid];

    // J-alpha
    auto c = area * inv_rest_area - alpha + act;

    // dJdx
    n = normalize(n);
    auto dcdq = cross(x20, n) * inv_rest_area * 0.5f;
    auto dcdr = cross(n, x10) * inv_rest_area * 0.5f;

    auto f_area = k_lambda * c;

    // -----------------------------
    // Area Damping

    auto dcdt = dot(dcdq, v1) + dot(dcdr, v2) - dot(dcdq + dcdr, v0);
    auto f_damp = k_damp * dcdt;

    f1 = f1 + dcdq * (f_area + f_damp);
    f2 = f2 + dcdr * (f_area + f_damp);
    auto f0 = f1 + f2;

    // -----------------------------
    // Lift + Drag

    auto vmid = (v0 + v1 + v2) * 0.3333f;
    auto vdir = normalize(vmid);

    auto f_drag = vmid * (k_drag * area * abs(dot(n, vmid)));
    auto f_lift = n * (k_lift * area * (1.57079f - acos(dot(n, vdir)))) * dot(vmid, vmid);

    // note reversed sign due to atomic_add below.. need to write the unary op -
    f0 = f0 - f_drag - f_lift;
    f1 = f1 + f_drag + f_lift;
    f2 = f2 + f_drag + f_lift;

    // apply forces
    atomic_add(f, i, f0);
    atomic_sub(f, j, f1);
    atomic_sub(f, k, f2);
}

__global__ void eval_triangles_contact(array_t<int> idx,
                                       int num_particles,
                                       array_t<vec3f> x,
                                       array_t<vec3f> v,
                                       array_t<int> indices,
                                       array_t<mat22f> pose,
                                       array_t<float> activation,
                                       array_t<float> materials,
                                       array_t<vec3f> f) {
    auto tid = wp::tid();
    auto face_no = tid / num_particles;    // which face
    auto particle_no = tid % num_particles;// which particle

    auto k_mu = materials(face_no, 0);
    auto k_lambda = materials(face_no, 1);
    auto k_damp = materials(face_no, 2);
    auto k_drag = materials(face_no, 3);
    auto k_lift = materials(face_no, 4);

    // at the moment, just one particle
    auto pos = x[particle_no];

    auto i = indices(face_no, 0);
    auto j = indices(face_no, 1);
    auto k = indices(face_no, 2);

    if (i == particle_no || j == particle_no || k == particle_no) {
        return;
    }

    auto p = x[i];// point zero
    auto q = x[j];// point one
    auto r = x[k];// point two

    // vp = v[i]; // vel zero
    // vq = v[j]; // vel one
    // vr = v[k]; // vel two
    //
    // qp = q-p; // barycentric coordinates (centered at p)
    // rp = r-p;

    auto bary = triangle_closest_point_barycentric(p, q, r, pos);
    auto closest = p * bary[0] + q * bary[1] + r * bary[2];

    auto diff = pos - closest;
    auto dist = dot(diff, diff);
    auto n = normalize(diff);
    auto c = min(dist - 0.01f, 0.0f);// 0 unless within 0.01 of surface
    // c = leaky_min(dot(n, x0)-0.01, 0.0, 0.0);
    auto fn = n * c * 1e5f;

    atomic_sub(f, particle_no, fn);

    // // apply forces (could do - f / 3 here)
    atomic_add(f, i, fn * bary[0]);
    atomic_add(f, j, fn * bary[1]);
    atomic_add(f, k, fn * bary[2]);
}

__global__ void eval_triangles_body_contacts(int num_particles,
                                             array_t<vec3f> x,
                                             array_t<vec3f> v,
                                             array_t<int> indices,
                                             array_t<vec3f> body_x,
                                             array_t<quatf> body_r,
                                             array_t<vec3f> body_v,
                                             array_t<vec3f> body_w,
                                             array_t<int> contact_body,
                                             array_t<vec3f> contact_point,
                                             array_t<float> contact_dist,
                                             array_t<int> contact_mat,
                                             array_t<float> materials,
                                             array_t<vec3f> tri_f) {
    auto tid = wp::tid();

    auto face_no = tid / num_particles;    // which face
    auto particle_no = tid % num_particles;// which particle

    // -----------------------
    // load body body point
    auto c_body = contact_body[particle_no];
    auto c_point = contact_point[particle_no];
    auto c_dist = contact_dist[particle_no];
    auto c_mat = contact_mat[particle_no];

    // hard coded surface parameter tensor layout (ke, kd, kf, mu)
    auto ke = materials[c_mat * 4 + 0];// restitution coefficient
    auto kd = materials[c_mat * 4 + 1];// damping coefficient
    auto kf = materials[c_mat * 4 + 2];// friction coefficient
    auto mu = materials[c_mat * 4 + 3];// coulomb friction

    auto x0 = body_x[c_body];// position of colliding body
    auto r0 = body_r[c_body];// orientation of colliding body

    auto v0 = body_v[c_body];
    auto w0 = body_w[c_body];

    // transform point to world space
    auto pos = x0 + quat_rotate(r0, c_point);
    // use x0 as center, everything is offset from center of mass

    // moment arm
    auto r = pos - x0;// basically just c_point in the new coordinates
    auto rhat = normalize(r);
    pos = pos + rhat * c_dist;// add on 'thickness' of shape, e.g.: radius of sphere/capsule

    // contact point velocity
    auto dpdt = v0 + cross(w0, r);// this is body velocity cross offset, so it's the velocity of the contact point.

    // -----------------------
    // load triangle
    auto i = indices[face_no * 3 + 0];
    auto j = indices[face_no * 3 + 1];
    auto k = indices[face_no * 3 + 2];

    auto p = x[i];// point zero
    auto q = x[j];// point one
    r = x[k];     // point two

    auto vp = v[i];// vel zero
    auto vq = v[j];// vel one
    auto vr = v[k];// vel two

    auto bary = triangle_closest_point_barycentric(p, q, r, pos);
    auto closest = p * bary[0] + q * bary[1] + r * bary[2];

    auto diff = pos - closest;      // vector from tri to point
    auto dist = dot(diff, diff);    // squared distance
    auto n = normalize(diff);       // points into the object
    auto c = min(dist - 0.05f, 0.f);// 0 unless within 0.05 of surface
    // c = leaky_min(dot(n, x0)-0.01, 0.0, 0.0);
    // fn = n * c * 1e6;    // points towards cloth (both n and c are negative)
    //
    // atomic_sub(tri_f, particle_no, fn);

    auto fn = c * ke;// normal force (restitution coefficient * how far inside for ground) (negative)

    auto vtri = vp * bary[0] + vq * bary[1] + vr * bary[2];// bad approximation for centroid velocity
    auto vrel = vtri - dpdt;

    auto vn = dot(n, vrel); // velocity component of body in negative normal direction
    auto vt = vrel - n * vn;// velocity component not in normal direction

    // contact damping
    auto fd = 0.f - max(vn, 0.f) * kd * step(c);// again, negative, into the ground

    // // viscous friction
    // ft = vt*kf;
    //
    // Coulomb friction (box)
    auto lower = mu * (fn + fd);
    auto upper = 0.f - lower;// workaround because no unary ops yet

    auto nx = cross(n, vec3(0.0, 0.0, 1.0));// basis vectors for tangent
    auto nz = cross(n, vec3(1.0, 0.0, 0.0));

    auto vx = clamp(dot(nx * kf, vt), lower, upper);
    auto vz = clamp(dot(nz * kf, vt), lower, upper);

    auto ft = (nx * vx + nz * vz) * (0.f - step(c));// vec3(vx, 0.0, vz)*step(c)

    // // Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    // ft = normalize(vt)*min(kf*length(vt), 0.0 - mu*c*ke);

    auto f_total = n * (fn + fd) + ft;

    atomic_add(tri_f, i, f_total * bary[0]);
    atomic_add(tri_f, j, f_total * bary[1]);
    atomic_add(tri_f, k, f_total * bary[2]);
}

__global__ void eval_bending(array_t<vec3f> x,
                             array_t<vec3f> v,
                             array_t<int> indices,
                             array_t<float> rest,
                             array_t<float> bending_properties,
                             array_t<vec3f> f) {
    auto tid = wp::tid();
    auto ke = bending_properties(tid, 0);
    auto kd = bending_properties(tid, 1);

    auto i = indices(tid, 0);
    auto j = indices(tid, 1);
    auto k = indices(tid, 2);
    auto l = indices(tid, 3);

    auto rest_angle = rest[tid];

    auto x1 = x[i];
    auto x2 = x[j];
    auto x3 = x[k];
    auto x4 = x[l];

    auto v1 = v[i];
    auto v2 = v[j];
    auto v3 = v[k];
    auto v4 = v[l];

    auto n1 = cross(x3 - x1, x4 - x1);// normal to face 1
    auto n2 = cross(x4 - x2, x3 - x2);// normal to face 2

    auto n1_length = length(n1);
    auto n2_length = length(n2);

    if (n1_length < 1.0e-6 || n2_length < 1.0e-6) {
        return;
    }
    auto rcp_n1 = 1.f / n1_length;
    auto rcp_n2 = 1.f / n2_length;

    auto cos_theta = dot(n1, n2) * rcp_n1 * rcp_n2;

    n1 = n1 * rcp_n1 * rcp_n1;
    n2 = n2 * rcp_n2 * rcp_n2;

    auto e = x4 - x3;
    auto e_hat = normalize(e);
    auto e_length = length(e);

    auto s = sign(dot(cross(n2, n1), e_hat));
    auto angle = acos(cos_theta) * s;

    auto d1 = n1 * e_length;
    auto d2 = n2 * e_length;
    auto d3 = n1 * dot(x1 - x4, e_hat) + n2 * dot(x2 - x4, e_hat);
    auto d4 = n1 * dot(x3 - x1, e_hat) + n2 * dot(x3 - x2, e_hat);

    // elastic
    auto f_elastic = ke * (angle - rest_angle);

    // damping
    auto f_damp = kd * (dot(d1, v1) + dot(d2, v2) + dot(d3, v3) + dot(d4, v4));

    // total force, proportional to edge length
    auto f_total = 0.f - e_length * (f_elastic + f_damp);

    atomic_add(f, i, d1 * f_total);
    atomic_add(f, j, d2 * f_total);
    atomic_add(f, k, d3 * f_total);
    atomic_add(f, l, d4 * f_total);
}

__global__ void eval_tetrahedra(array_t<vec3f> x,
                                array_t<vec3f> v,
                                array_t<int> indices,
                                array_t<mat33f> pose,
                                array_t<float> activation,
                                array_t<float> materials,
                                array_t<vec3f> f) {
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

    auto alpha = 1.f + k_mu / k_lambda - k_mu / (4.f * k_lambda);

    // scale stiffness coefficients to account for area
    k_mu = k_mu * rest_volume;
    k_lambda = k_lambda * rest_volume;
    k_damp = k_damp * rest_volume;

    // F = Xs*Xm^-1
    auto F = Ds * Dm;
    auto dFdt = mat33(v10, v20, v30) * Dm;

    auto col1 = vec3(F(0, 0), F(1, 0), F(2, 0));
    auto col2 = vec3(F(0, 1), F(1, 1), F(2, 1));
    auto col3 = vec3(F(0, 2), F(1, 2), F(2, 2));

    // -----------------------------
    // Neo-Hookean (with rest stability [Smith et al 2018])

    auto Ic = dot(col1, col1) + dot(col2, col2) + dot(col3, col3);

    // deviatoric part
    auto P = F * k_mu * (1.f - 1.f / (Ic + 1.f)) + dFdt * k_damp;
    auto H = P * transpose(Dm);

    auto f1 = vec3(H(0, 0), H(1, 0), H(2, 0));
    auto f2 = vec3(H(0, 1), H(1, 1), H(2, 1));
    auto f3 = vec3(H(0, 2), H(1, 2), H(2, 2));

    // -----------------------------
    // C_sqrt
    //
    // alpha = 1.0/
    //
    // r_s = sqrt(abs(dot(col1, col1) + dot(col2, col2) + dot(col3, col3) - 3.0))/
    //
    // f1 = vec3()/
    // f2 = vec3()/
    // f3 = vec3()/
    //
    // if (r_s > 0.0) {
    // auto  r_s_inv = 1.f/ r_s;
    //
    //  auto  C = r_s;
    //  auto    dCdx = F*transpose(Dm)*r_s_inv*sign(r_s);
    //
    // auto   grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0]);
    //  auto   grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1]);
    // auto    grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2]);
    //
    // f1 = grad1*C*k_mu;
    // f2 = grad2*C*k_mu;
    // f3 = grad3*C*k_mu;
    //  }
    // -----------------------------
    // C_spherical
    //
    // alpha = 1.0;
    //
    // r_s = sqrt(dot(col1, col1) + dot(col2, col2) + dot(col3, col3));
    // r_s_inv = 1.0/r_s;
    //
    // C = r_s - sqrt(3.0);
    // dCdx = F*transpose(Dm)*r_s_inv;
    //
    // grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0]);
    // grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1]);
    // grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2]);
    //
    // f1 = grad1*C*k_mu;
    // f2 = grad2*C*k_mu;
    // f3 = grad3*C*k_mu;
    //
    // ----------------------------
    // C_D
    //
    // alpha = 1.0;
    //
    // r_s = sqrt(dot(col1, col1) + dot(col2, col2) + dot(col3, col3));
    //
    // C = r_s*r_s - 3.0;
    // dCdx = F*transpose(Dm)*2.0;
    //
    // grad1 = vec3(dCdx[0,0], dCdx[1,0], dCdx[2,0]);
    // grad2 = vec3(dCdx[0,1], dCdx[1,1], dCdx[2,1]);
    // grad3 = vec3(dCdx[0,2], dCdx[1,2], dCdx[2,2]);
    //
    // f1 = grad1*C*k_mu;
    // f2 = grad2*C*k_mu;
    // f3 = grad3*C*k_mu;
    //
    // ----------------------------
    // Hookean
    //
    // alpha = 1.0;
    //
    // I = mat33(vec3(1.0, 0.0, 0.0),
    //`          vec3(0.0, 1.0, 0.0),
    //`          vec3(0.0, 0.0, 1.0));
    //
    // P = (F + transpose(F) + I*(0.0-2.0))*k_mu;
    // H = P * transpose(Dm);
    //
    // f1 = vec3(H[0, 0], H[1, 0], H[2, 0]);
    // f2 = vec3(H[0, 1], H[1, 1], H[2, 1]);
    // f3 = vec3(H[0, 2], H[1, 2], H[2, 2]);
    //
    // hydrostatic part
    auto J = determinant(F);

    // print(J)
    auto s = inv_rest_volume / 6.f;
    auto dJdx1 = cross(x20, x30) * s;
    auto dJdx2 = cross(x30, x10) * s;
    auto dJdx3 = cross(x10, x20) * s;

    auto f_volume = (J - alpha + act) * k_lambda;
    auto f_damp = (dot(dJdx1, v1) + dot(dJdx2, v2) + dot(dJdx3, v3)) * k_damp;

    auto f_total = f_volume + f_damp;

    f1 = f1 + dJdx1 * f_total;
    f2 = f2 + dJdx2 * f_total;
    f3 = f3 + dJdx3 * f_total;
    auto f0 = (f1 + f2 + f3) * (0.f - 1.f);

    // apply forces
    atomic_sub(f, i, f0);
    atomic_sub(f, j, f1);
    atomic_sub(f, k, f2);
    atomic_sub(f, l, f3);
}

__global__ void eval_particle_ground_contacts(array_t<vec3f> particle_x,
                                              array_t<vec3f> particle_v,
                                              array_t<float> particle_radius,
                                              array_t<uint32_t> particle_flags,
                                              float ke,
                                              float kd,
                                              float kf,
                                              float mu,
                                              array_t<float> ground,
                                              array_t<vec3f> f) {
    auto tid = wp::tid();
    if ((particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0) {
        return;
    }
    auto x = particle_x[tid];
    auto v = particle_v[tid];
    auto radius = particle_radius[tid];

    auto n = vec3(ground[0], ground[1], ground[2]);
    auto c = min(dot(n, x) + ground[3] - radius, 0.f);

    auto vn = dot(n, v);
    auto jn = c * ke;

    if (c >= 0.0) {
        return;
    }
    auto jd = min(vn, 0.0f) * kd;

    // contact force
    auto fn = jn + jd;

    // friction force
    auto vt = v - n * vn;
    auto vs = length(vt);

    if (vs > 0.0) {
        vt = vt / vs;
    }
    // Coulomb condition
    auto ft = min(vs * kf, mu * abs(fn));

    // total force
    f[tid] = f[tid] - n * fn - vt * ft;
}

__global__ void eval_particle_contacts(array_t<vec3f> particle_x,
                                       array_t<vec3f> particle_v,
                                       array_t<transformf> body_q,
                                       array_t<spatial_vectorf> body_qd,
                                       array_t<float> particle_radius,
                                       array_t<uint32_t> particle_flags,
                                       array_t<vec3f> body_com,
                                       array_t<int> shape_body,
                                       ModelShapeMaterials shape_materials,
                                       float particle_ke,
                                       float particle_kd,
                                       float particle_kf,
                                       float particle_mu,
                                       float particle_ka,
                                       array_t<int> contact_count,
                                       array_t<int> contact_particle,
                                       array_t<int> contact_shape,
                                       array_t<vec3f> contact_body_pos,
                                       array_t<vec3f> contact_body_vel,
                                       array_t<vec3f> contact_normal,
                                       int contact_max,
                                       array_t<vec3f> particle_f,
                                       array_t<spatial_vectorf> body_f) {
    auto tid = wp::tid();

    auto count = min(contact_max, contact_count[0]);
    if (tid >= count) {
        return;
    }
    auto shape_index = contact_shape[tid];
    auto body_index = shape_body[shape_index];
    auto particle_index = contact_particle[tid];
    if ((particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0) {
        return;
    }
    auto px = particle_x[particle_index];
    auto pv = particle_v[particle_index];

    auto X_wb = transform_identity();
    auto X_com = vec3();
    auto body_v_s = spatial_vector();

    if (body_index >= 0) {
        X_wb = body_q[body_index];
        X_com = body_com[body_index];
        body_v_s = body_qd[body_index];
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
    auto ke = 0.5f * (particle_ke + shape_materials.ke[shape_index]);
    auto kd = 0.5f * (particle_kd + shape_materials.kd[shape_index]);
    auto kf = 0.5f * (particle_kf + shape_materials.kf[shape_index]);
    auto mu = 0.5f * (particle_mu + shape_materials.mu[shape_index]);

    auto body_w = spatial_top(body_v_s);
    auto body_v = spatial_bottom(body_v_s);

    // compute the body velocity at the particle position
    auto bv = body_v + cross(body_w, r) + transform_vector(X_wb, contact_body_vel[tid]);

    // relative velocity
    auto v = pv - bv;

    // decompose relative velocity
    auto vn = dot(n, v);
    auto vt = v - n * vn;

    // contact elastic
    auto fn = n * c * ke;

    // contact damping
    auto fd = n * min(vn, 0.f) * kd;

    // viscous friction
    // ft = vt*kf;

    // Coulomb friction (box)
    // lower = mu * c * ke;
    // upper = 0.0 - lower;

    // vx = clamp(dot(vec3(kf, 0.0, 0.0), vt), lower, upper);
    // vz = clamp(dot(vec3(0.0, 0.0, kf), vt), lower, upper);

    // ft = vec3(vx, 0.0, vz);

    // Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    auto ft = normalize(vt) * min(kf * length(vt), abs(mu * c * ke));

    auto f_total = fn + (fd + ft);
    auto t_total = cross(r, f_total);

    atomic_sub(particle_f, particle_index, f_total);

    if (body_index >= 0) {
        atomic_add(body_f, body_index, spatial_vector(t_total, f_total));
    }
}

__global__ void eval_rigid_contacts(array_t<transformf> body_q,
                                    array_t<spatial_vectorf> body_qd,
                                    array_t<vec3f> body_com,
                                    ModelShapeMaterials shape_materials,
                                    ModelShapeGeometry geo,
                                    array_t<int> contact_count,
                                    array_t<int> contact_body0,
                                    array_t<int> contact_body1,
                                    array_t<vec3f> contact_point0,
                                    array_t<vec3f> contact_point1,
                                    array_t<vec3f> contact_normal,
                                    array_t<int> contact_shape0,
                                    array_t<int> contact_shape1,
                                    array_t<spatial_vectorf> body_f) {
    auto tid = wp::tid();
    if (contact_shape0[tid] == contact_shape1[tid]) {
        return;
    }

    auto count = contact_count[0];
    if (tid >= count) {
        return;
    }

    // retrieve contact thickness, compute average contact material properties
    auto ke = 0.f;// restitution coefficient
    auto kd = 0.f;// damping coefficient
    auto kf = 0.f;// friction coefficient
    auto mu = 0.f;// coulomb friction
    auto mat_nonzero = 0;
    auto thickness_a = 0.f;
    auto thickness_b = 0.f;
    auto shape_a = contact_shape0[tid];
    auto shape_b = contact_shape1[tid];
    if (shape_a >= 0) {
        mat_nonzero += 1;
        ke += shape_materials.ke[shape_a];
        kd += shape_materials.kd[shape_a];
        kf += shape_materials.kf[shape_a];
        mu += shape_materials.mu[shape_a];
        thickness_a = geo.thickness[shape_a];
    }
    if (shape_b >= 0) {
        mat_nonzero += 1;
        ke += shape_materials.ke[shape_b];
        kd += shape_materials.kd[shape_b];
        kf += shape_materials.kf[shape_b];
        mu += shape_materials.mu[shape_b];
        thickness_b = geo.thickness[shape_b];
    }
    if (mat_nonzero > 0) {
        ke = ke / float(mat_nonzero);
        kd = kd / float(mat_nonzero);
        kf = kf / float(mat_nonzero);
        mu = mu / float(mat_nonzero);
    }

    auto body_a = contact_body0[tid];
    auto body_b = contact_body1[tid];

    // body position in world space
    auto n = contact_normal[tid];
    auto bx_a = contact_point0[tid];
    auto bx_b = contact_point1[tid];
    vec3 r_a, r_b;
    if (body_a >= 0) {
        auto X_wb_a = body_q[body_a];
        auto X_com_a = body_com[body_a];
        bx_a = transform_point(X_wb_a, bx_a) - thickness_a * n;
        r_a = bx_a - transform_point(X_wb_a, X_com_a);
    }
    if (body_b >= 0) {
        auto X_wb_b = body_q[body_b];
        auto X_com_b = body_com[body_b];
        bx_b = transform_point(X_wb_b, bx_b) + thickness_b * n;
        r_b = bx_b - transform_point(X_wb_b, X_com_b);
    }
    auto d = dot(n, bx_a - bx_b);

    if (d >= 0.0) {
        return;
    }

    // compute contact point velocity
    auto bv_a = vec3(0.0);
    auto bv_b = vec3(0.0);
    if (body_a >= 0) {
        auto body_v_s_a = body_qd[body_a];
        auto body_w_a = spatial_top(body_v_s_a);
        auto body_v_a = spatial_bottom(body_v_s_a);
        bv_a = body_v_a + cross(body_w_a, r_a);
    }
    if (body_b >= 0) {
        auto body_v_s_b = body_qd[body_b];
        auto body_w_b = spatial_top(body_v_s_b);
        auto body_v_b = spatial_bottom(body_v_s_b);
        bv_b = body_v_b + cross(body_w_b, r_b);
    }
    // relative velocity
    auto v = bv_a - bv_b;

    // print(v);

    // decompose relative velocity
    auto vn = dot(n, v);
    auto vt = v - n * vn;

    // contact elastic
    auto fn = d * ke;

    // contact damping
    auto fd = min(vn, 0.f) * kd * step(d);

    // viscous friction
    // ft = vt*kf;

    // Coulomb friction (box)
    // lower = mu * d * ke;
    // upper = 0.0 - lower;

    // vx = clamp(dot(vec3(kf, 0.0, 0.0), vt), lower, upper);
    // vz = clamp(dot(vec3(0.0, 0.0, kf), vt), lower, upper);

    // ft = vec3(vx, 0.0, vz);

    // Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    // ft = normalize(vt)*min(kf*length(vt), abs(mu*d*ke));
    auto ft = normalize(vt) * min(kf * length(vt), 0.f - mu * (fn + fd));

    // f_total = fn + (fd + ft);
    auto f_total = n * (fn + fd) + ft;
    // t_total = cross(r, f_total);

    // print("apply contact force");
    // print(f_total);

    if (body_a >= 0) {
        atomic_sub(body_f, body_a, spatial_vector(cross(r_a, f_total), f_total));
    }
    if (body_b >= 0) {
        atomic_add(body_f, body_b, spatial_vector(cross(r_b, f_total), f_total));
    }
}

CUDA_CALLABLE_DEVICE vec3f eval_joint_force(float q,
                                            float qd,
                                            float target,
                                            float target_ke,
                                            float target_kd,
                                            float act,
                                            float limit_lower,
                                            float limit_upper,
                                            float limit_ke,
                                            float limit_kd,
                                            vec3f axis) {
    auto limit_f = 0.f;

    // compute limit forces, damping only active when limit is violated
    if (q < limit_lower) {
        limit_f = limit_ke * (limit_lower - q) - limit_kd * min(qd, 0.f);
    }
    if (q > limit_upper) {
        limit_f = limit_ke * (limit_upper - q) - limit_kd * max(qd, 0.f);
    }
    // joint dynamics
    auto total_f = (target_ke * (q - target) + target_kd * qd + act - limit_f) * axis;

    return total_f;
}

__global__ void eval_body_joints(array_t<transformf> body_q,
                                 array_t<spatial_vectorf> body_qd,
                                 array_t<vec3f> body_com,
                                 array_t<int> joint_q_start,
                                 array_t<int> joint_qd_start,
                                 array_t<int> joint_type,
                                 array_t<int> joint_enabled,
                                 array_t<int> joint_child,
                                 array_t<int> joint_parent,
                                 array_t<transformf> joint_X_p,
                                 array_t<transformf> joint_X_c,
                                 array_t<vec3f> joint_axis,
                                 array_t<int> joint_axis_start,
                                 array_t<int> joint_axis_dim,
                                 array_t<float> joint_target,
                                 array_t<float> joint_act,
                                 array_t<float> joint_target_ke,
                                 array_t<float> joint_target_kd,
                                 array_t<float> joint_limit_lower,
                                 array_t<float> joint_limit_upper,
                                 array_t<float> joint_limit_ke,
                                 array_t<float> joint_limit_kd,
                                 float joint_attach_ke,
                                 float joint_attach_kd,
                                 array_t<spatial_vectorf> body_f) {
    auto tid = wp::tid();
    auto type = joint_type[tid];

    // early out for free joints
    if (joint_enabled[tid] == 0 || type == int(JointType::JOINT_FREE)) {
        return;
    }

    auto c_child = joint_child[tid];
    auto c_parent = joint_parent[tid];

    auto X_pj = joint_X_p[tid];
    auto X_cj = joint_X_c[tid];

    auto X_wp = X_pj;
    auto r_p = vec3();
    auto w_p = vec3();
    auto v_p = vec3();

    // parent transform and moment arm
    if (c_parent >= 0) {
        X_wp = body_q[c_parent] * X_wp;
        r_p = transform_get_translation(X_wp) - transform_point(body_q[c_parent], body_com[c_parent]);

        auto twist_p = body_qd[c_parent];

        w_p = spatial_top(twist_p);
        v_p = spatial_bottom(twist_p) + cross(w_p, r_p);
    }
    // child transform and moment arm
    auto X_wc = body_q[c_child] * X_cj;
    auto r_c = transform_get_translation(X_wc) - transform_point(body_q[c_child], body_com[c_child]);

    auto twist_c = body_qd[c_child];

    auto w_c = spatial_top(twist_c);
    auto v_c = spatial_bottom(twist_c) + cross(w_c, r_c);

    // joint properties (for 1D joints)
    auto q_start = joint_q_start[tid];
    auto qd_start = joint_qd_start[tid];
    auto axis_start = joint_axis_start[tid];

    auto target = joint_target[axis_start];
    auto target_ke = joint_target_ke[axis_start];
    auto target_kd = joint_target_kd[axis_start];
    auto limit_ke = joint_limit_ke[axis_start];
    auto limit_kd = joint_limit_kd[axis_start];
    auto limit_lower = joint_limit_lower[axis_start];
    auto limit_upper = joint_limit_upper[axis_start];

    auto act = joint_act[qd_start];

    auto x_p = transform_get_translation(X_wp);
    auto x_c = transform_get_translation(X_wc);

    auto q_p = transform_get_rotation(X_wp);
    auto q_c = transform_get_rotation(X_wc);

    // translational error
    auto x_err = x_c - x_p;
    auto r_err = quat_inverse(q_p) * q_c;
    auto v_err = v_c - v_p;
    auto w_err = w_c - w_p;

    // total force/torque on the parent
    auto t_total = vec3();
    auto f_total = vec3();

    // reduce angular damping stiffness for stability
    auto angular_damping_scale = 0.01f;

    if (type == int(JointType::JOINT_FIXED)) {
        auto ang_err = normalize(vec3(r_err[0], r_err[1], r_err[2])) * acos(r_err[3]) * 2.f;

        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd;
        t_total += (transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale);
    }

    if (type == int(JointType::JOINT_PRISMATIC)) {
        auto axis = joint_axis[axis_start];

        // world space joint axis
        auto axis_p = transform_vector(X_wp, axis);

        // evaluate joint coordinates
        auto q = dot(x_err, axis_p);
        auto qd = dot(v_err, axis_p);

        f_total = eval_joint_force(
            q, qd, target, target_ke, target_kd, act, limit_lower, limit_upper, limit_ke, limit_kd, axis_p);

        // attachment dynamics
        auto ang_err = normalize(vec3(r_err[0], r_err[1], r_err[2])) * acos(r_err[3]) * 2.f;

        // project off any displacement along the joint axis
        f_total += (x_err - q * axis_p) * joint_attach_ke + (v_err - qd * axis_p) * joint_attach_kd;
        t_total += (transform_vector(X_wp, ang_err) * joint_attach_ke + w_err * joint_attach_kd * angular_damping_scale);
    }

    if (type == int(JointType::JOINT_REVOLUTE)) {
        auto axis = joint_axis[axis_start];

        auto axis_p = transform_vector(X_wp, axis);
        auto axis_c = transform_vector(X_wc, axis);

        // swing twist decomposition
        auto twist = quat_twist(axis, r_err);

        auto q = acos(twist[3]) * 2.f * sign(dot(axis, vec3(twist[0], twist[1], twist[2])));
        auto qd = dot(w_err, axis_p);

        t_total = eval_joint_force(
            q, qd, target, target_ke, target_kd, act, limit_lower, limit_upper, limit_ke, limit_kd, axis_p);

        // attachment dynamics
        auto swing_err = cross(axis_p, axis_c);

        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd;
        t_total += swing_err * joint_attach_ke + (w_err - qd * axis_p) * joint_attach_kd * angular_damping_scale;
    }

    if (type == int(JointType::JOINT_BALL)) {
        auto ang_err = normalize(vec3(r_err[0], r_err[1], r_err[2])) * acos(r_err[3]) * 2.f;

        // todo: joint limits
        t_total += target_kd * w_err + target_ke * transform_vector(X_wp, ang_err);
        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd;
    }

    if (type == int(JointType::JOINT_COMPOUND)) {
        auto q_pc = quat_inverse(q_p) * q_c;

        // decompose to a compound rotation each axis
        auto angles = quat_decompose(q_pc);

        // reconstruct rotation axes
        auto axis_0 = vec3(1.0, 0.0, 0.0);
        auto q_0 = quat_from_axis_angle(axis_0, angles[0]);

        auto axis_1 = quat_rotate(q_0, vec3(0.0, 1.0, 0.0));
        auto q_1 = quat_from_axis_angle(axis_1, angles[1]);

        auto axis_2 = quat_rotate(q_1 * q_0, vec3(0.0, 0.0, 1.0));
        auto q_2 = quat_from_axis_angle(axis_2, angles[2]);

        auto q_w = q_p;

        axis_0 = transform_vector(X_wp, axis_0);
        axis_1 = transform_vector(X_wp, axis_1);
        axis_2 = transform_vector(X_wp, axis_2);

        // joint dynamics
        t_total = vec3();
        // // TODO remove quat_rotate(q_w, ...)?
        // t_total += eval_joint_force(angles[0], dot(quat_rotate(q_w, axis_0), w_err), joint_target[axis_start+0], joint_target_ke[axis_start+0],joint_target_kd[axis_start+0], joint_act[axis_start+0], joint_limit_lower[axis_start+0], joint_limit_upper[axis_start+0], joint_limit_ke[axis_start+0], joint_limit_kd[axis_start+0], quat_rotate(q_w, axis_0))
        // t_total += eval_joint_force(angles[1], dot(quat_rotate(q_w, axis_1), w_err), joint_target[axis_start+1], joint_target_ke[axis_start+1],joint_target_kd[axis_start+1], joint_act[axis_start+1], joint_limit_lower[axis_start+1], joint_limit_upper[axis_start+1], joint_limit_ke[axis_start+1], joint_limit_kd[axis_start+1], quat_rotate(q_w, axis_1))
        // t_total += eval_joint_force(angles[2], dot(quat_rotate(q_w, axis_2), w_err), joint_target[axis_start+2], joint_target_ke[axis_start+2],joint_target_kd[axis_start+2], joint_act[axis_start+2], joint_limit_lower[axis_start+2], joint_limit_upper[axis_start+2], joint_limit_ke[axis_start+2], joint_limit_kd[axis_start+2], quat_rotate(q_w, axis_2))

        t_total += eval_joint_force(
            angles[0],
            dot(axis_0, w_err),
            joint_target[axis_start + 0],
            joint_target_ke[axis_start + 0],
            joint_target_kd[axis_start + 0],
            joint_act[axis_start + 0],
            joint_limit_lower[axis_start + 0],
            joint_limit_upper[axis_start + 0],
            joint_limit_ke[axis_start + 0],
            joint_limit_kd[axis_start + 0],
            axis_0);
        t_total += eval_joint_force(
            angles[1],
            dot(axis_1, w_err),
            joint_target[axis_start + 1],
            joint_target_ke[axis_start + 1],
            joint_target_kd[axis_start + 1],
            joint_act[axis_start + 1],
            joint_limit_lower[axis_start + 1],
            joint_limit_upper[axis_start + 1],
            joint_limit_ke[axis_start + 1],
            joint_limit_kd[axis_start + 1],
            axis_1);
        t_total += eval_joint_force(
            angles[2],
            dot(axis_2, w_err),
            joint_target[axis_start + 2],
            joint_target_ke[axis_start + 2],
            joint_target_kd[axis_start + 2],
            joint_act[axis_start + 2],
            joint_limit_lower[axis_start + 2],
            joint_limit_upper[axis_start + 2],
            joint_limit_ke[axis_start + 2],
            joint_limit_kd[axis_start + 2],
            axis_2);

        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd;
    }

    if (type == int(JointType::JOINT_UNIVERSAL)) {
        auto q_pc = quat_inverse(q_p) * q_c;

        // decompose to a compound rotation each axis
        auto angles = quat_decompose(q_pc);

        // reconstruct rotation axes
        auto axis_0 = vec3(1.0, 0.0, 0.0);
        auto q_0 = quat_from_axis_angle(axis_0, angles[0]);

        auto axis_1 = quat_rotate(q_0, vec3(0.0, 1.0, 0.0));
        auto q_1 = quat_from_axis_angle(axis_1, angles[1]);

        auto axis_2 = quat_rotate(q_1 * q_0, vec3(0.0, 0.0, 1.0));
        auto q_2 = quat_from_axis_angle(axis_2, angles[2]);

        auto q_w = q_p;

        axis_0 = transform_vector(X_wp, axis_0);
        axis_1 = transform_vector(X_wp, axis_1);
        axis_2 = transform_vector(X_wp, axis_2);

        // joint dynamics
        t_total = vec3();

        // free axes
        // // TODO remove quat_rotate(q_w, ...)?
        // t_total += eval_joint_force(angles[0], dot(quat_rotate(q_w, axis_0), w_err), joint_target[axis_start+0], joint_target_ke[axis_start+0],joint_target_kd[axis_start+0], joint_act[axis_start+0], joint_limit_lower[axis_start+0], joint_limit_upper[axis_start+0], joint_limit_ke[axis_start+0], joint_limit_kd[axis_start+0], quat_rotate(q_w, axis_0))
        // t_total += eval_joint_force(angles[1], dot(quat_rotate(q_w, axis_1), w_err), joint_target[axis_start+1], joint_target_ke[axis_start+1],joint_target_kd[axis_start+1], joint_act[axis_start+1], joint_limit_lower[axis_start+1], joint_limit_upper[axis_start+1], joint_limit_ke[axis_start+1], joint_limit_kd[axis_start+1], quat_rotate(q_w, axis_1))

        // // last axis (fixed)
        // t_total += eval_joint_force(angles[2], dot(quat_rotate(q_w, axis_2), w_err), 0.0, joint_attach_ke, joint_attach_kd*angular_damping_scale, 0.0, 0.0, 0.0, 0.0, 0.0, quat_rotate(q_w, axis_2))

        // TODO remove quat_rotate(q_w, ...)?
        t_total += eval_joint_force(
            angles[0],
            dot(axis_0, w_err),
            joint_target[axis_start + 0],
            joint_target_ke[axis_start + 0],
            joint_target_kd[axis_start + 0],
            joint_act[axis_start + 0],
            joint_limit_lower[axis_start + 0],
            joint_limit_upper[axis_start + 0],
            joint_limit_ke[axis_start + 0],
            joint_limit_kd[axis_start + 0],
            axis_0);
        t_total += eval_joint_force(
            angles[1],
            dot(axis_1, w_err),
            joint_target[axis_start + 1],
            joint_target_ke[axis_start + 1],
            joint_target_kd[axis_start + 1],
            joint_act[axis_start + 1],
            joint_limit_lower[axis_start + 1],
            joint_limit_upper[axis_start + 1],
            joint_limit_ke[axis_start + 1],
            joint_limit_kd[axis_start + 1],
            axis_1);

        // last axis (fixed)
        t_total += eval_joint_force(
            angles[2],
            dot(axis_2, w_err),
            0.0,
            joint_attach_ke,
            joint_attach_kd * angular_damping_scale,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            axis_2);

        f_total += x_err * joint_attach_ke + v_err * joint_attach_kd;
    }

    // write forces
    if (c_parent >= 0) {
        atomic_add(body_f, c_parent, spatial_vector(t_total + cross(r_p, f_total), f_total));
    }
    atomic_sub(body_f, c_child, spatial_vector(t_total + cross(r_c, f_total), f_total));
}

CUDA_CALLABLE_DEVICE vec3f compute_muscle_force(int i, array_t<transformf> body_X_s,
                                                array_t<spatial_vectorf> body_v_s,
                                                array_t<vec3f> body_com,
                                                array_t<int> muscle_links,
                                                array_t<vec3f> muscle_points,
                                                float muscle_activation,
                                                array_t<spatial_vectorf> body_f_s) {
    auto link_0 = muscle_links[i];
    auto link_1 = muscle_links[i + 1];

    if (link_0 == link_1) {
        return 0;
    }

    auto r_0 = muscle_points[i];
    auto r_1 = muscle_points[i + 1];

    auto xform_0 = body_X_s[link_0];
    auto xform_1 = body_X_s[link_1];

    auto pos_0 = transform_point(xform_0, r_0 - body_com[link_0]);
    auto pos_1 = transform_point(xform_1, r_1 - body_com[link_1]);

    auto n = normalize(pos_1 - pos_0);

    // todo: add passive elastic and viscosity terms
    auto f = n * muscle_activation;

    atomic_sub(body_f_s, link_0, spatial_vector(f, cross(pos_0, f)));
    atomic_add(body_f_s, link_1, spatial_vector(f, cross(pos_1, f)));

    return f;
}

__global__ void eval_muscles(array_t<transformf> body_X_s,
                             array_t<spatial_vectorf> body_v_s,
                             array_t<vec3f> body_com,
                             array_t<int> muscle_start,
                             array_t<float> muscle_params,
                             array_t<int> muscle_links,
                             array_t<vec3f> muscle_points,
                             array_t<float> muscle_activation,
                             array_t<spatial_vectorf> body_f_s) {
    auto tid = wp::tid();

    auto m_start = muscle_start[tid];
    auto m_end = muscle_start[tid + 1] - 1;

    auto activation = muscle_activation[tid];

    for (int i = m_start; i < m_end; i++) {
        compute_muscle_force(i, body_X_s, body_v_s, body_com, muscle_links, muscle_points, activation, body_f_s);
    }
}

void compute_forces(Model &model, State &state, array_t<float> particle_f, array_t<float> body_f) {}

SemiImplicitIntegrator::SemiImplicitIntegrator(float angular_damping)
    : angular_damping_{angular_damping} {
}

void SemiImplicitIntegrator::simulate(Model &model, State &state_in, State &state_out, float dt) {
}

__global__ void compute_particle_residual(array_t<vec3f> particle_qd_0,
                                          array_t<vec3f> particle_qd_1,
                                          array_t<vec3f> particle_f,
                                          array_t<float> particle_m,
                                          array_t<uint32_t> particle_flags,
                                          vec3f gravity,
                                          float dt,
                                          array_t<vec3f> residual) {
    auto tid = wp::tid();
    if ((particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0) {
        return;
    }
    auto m = particle_m[tid];
    auto v1 = particle_qd_1[tid];
    auto v0 = particle_qd_0[tid];
    auto f = particle_f[tid];

    auto err = vec3();

    if (m > 0.0) {
        err = (v1 - v0) * m - f * dt - gravity * dt * m;
    }
    residual[tid] = err;
}

__global__ void update_particle_position(array_t<vec3f> particle_q_0,
                                         array_t<vec3f> particle_q_1,
                                         array_t<vec3f> particle_qd_1,
                                         array_t<vec3f> x,
                                         array_t<uint32_t> particle_flags,
                                         float dt) {
    auto tid = wp::tid();
    if ((particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0) {
        return;
    }
    auto qd_1 = x[tid];

    auto q_0 = particle_q_0[tid];
    auto q_1 = q_0 + qd_1 * dt;

    particle_q_1[tid] = q_1;
    particle_qd_1[tid] = qd_1;
}

void compute_residual(Model &model, State &state_in, State &state_out, array_t<float> particle_f, float residual, float dt) {}

void init_state(Model &model, State &state_in, State &state_out, float dt) {}

void update_state(Model &model, State &state_in, State &state_out, array_t<float> x, float dt) {}

VariationalImplicitIntegrator::VariationalImplicitIntegrator(Model &model, int solver, float alpha, int max_iter, bool report) {
}

void VariationalImplicitIntegrator::simulate(Model &model, State &state_in, State &state_out, float dt) {
}
}// namespace wp