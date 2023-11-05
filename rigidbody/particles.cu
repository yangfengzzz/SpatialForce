//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "particles.h"
#include "core/hashgrid.h"

namespace wp {
CUDA_CALLABLE_DEVICE vec3f particle_force(vec3f n, vec3f v, float c, float k_n, float k_d, float k_f, float k_mu) {
    auto vn = dot(n, v);
    auto jn = c * k_n;
    auto jd = min(vn, 0.f) * k_d;

    // contact force
    auto fn = jn + jd;

    // friction force
    auto vt = v - n * vn;
    auto vs = length(vt);

    if (vs > 0.0) {
        vt = vt / vs;
    }
    // Coulomb condition
    auto ft = min(vs * k_f, k_mu * abs(fn));

    // total force
    return -n * fn - vt * ft;
}

__global__ void eval_particle_forces_kernel(uint64_t grid,
                                            const array_t<vec3f> &particle_x,
                                            const array_t<vec3f> &particle_v,
                                            const array_t<float> &particle_radius,
                                            const array_t<uint32_t> &particle_flags, float k_contact,
                                            float k_damp, float k_friction, float k_mu, float k_cohesion, float max_radius,
                                            array_t<vec3f> &particle_f) {
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

    auto f = vec3();

    // particle contact
    auto query = hash_grid_query(grid, x, radius + max_radius + k_cohesion);
    auto index = int(0);

    auto count = int(0);

    while (hash_grid_query_next(query, index)) {
        if ((particle_flags[index] & PARTICLE_FLAG_ACTIVE) != 0 && index != i) {
            // compute distance to point
            auto n = x - particle_x[index];
            auto d = length(n);
            auto err = d - radius - particle_radius[index];

            count += 1;

            if (err <= k_cohesion) {
                n = n / d;
                auto vrel = v - particle_v[index];

                f = f + particle_force(n, vrel, err, k_contact, k_damp, k_friction, k_mu);
            }
        }
    }
    particle_f[i] = f;
}

void eval_particle_forces(Model &model, State &state, array_t<vec3f> forces) {}
}// namespace wp
