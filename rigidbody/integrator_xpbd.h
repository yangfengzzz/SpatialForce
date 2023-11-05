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
/// A implicit integrator using XPBD
///
///    After constructing `Model` and `State` objects this time-integrator
///    may be used to advance the simulation state forward in time.
class XPBDIntegrator {
public:
    explicit XPBDIntegrator(int iterations = 2,
                            float soft_body_relaxation = 0.9,
                            float soft_contact_relaxation = 0.9,
                            float joint_linear_relaxation = 0.7,
                            float joint_angular_relaxation = 0.4,
                            float rigid_contact_relaxation = 0.8,
                            bool rigid_contact_con_weighting = true,
                            float angular_damping = 0.0,
                            bool enable_restitution = false);

    void simulate(Model &model, State &state_in, State &state_out, float dt);

private:
    int iterations_;

    float soft_body_relaxation_;
    float soft_contact_relaxation_;

    float joint_linear_relaxation_;
    float joint_angular_relaxation_;

    float rigid_contact_relaxation_;
    bool rigid_contact_con_weighting_;

    float angular_damping_;

    bool enable_restitution_;
};
}// namespace wp