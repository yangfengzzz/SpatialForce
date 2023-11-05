//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vector>

#include "core/array.h"
#include "core/mat.h"
#include "model.h"

namespace wp {
void compute_forces(Model &model, State &state, array_t<float> particle_f, array_t<float> body_f);

class SemiImplicitIntegrator {
public:
    explicit SemiImplicitIntegrator(float angular_damping = 0.05);

    void simulate(Model &model, State &state_in, State &state_out, float dt);

private:
    float angular_damping_;
};

void compute_residual(Model &model, State &state_in, State &state_out, array_t<float> particle_f, float residual, float dt);

void init_state(Model &model, State &state_in, State &state_out, float dt);

// compute the final positions given output velocity (x)
void update_state(Model &model, State &state_in, State &state_out, array_t<float> x, float dt);

class VariationalImplicitIntegrator {
public:
    VariationalImplicitIntegrator(Model &model, int solver, float alpha = 0.1, int max_iters = 32, bool report = false);

    void simulate(Model &model, State &state_in, State &state_out, float dt);

private:
};

}// namespace wp