//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/vec.h"
#include "model.h"

namespace wp {
void eval_particle_forces(Model &model, State &state, array_t<vec3f> forces);

}// namespace wp