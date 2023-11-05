//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/array.h"
#include "core/spatial.h"
#include "model.h"

namespace wp {
// updates state body information based on joint coordinates
void eval_fk(Model &model, array_t<float> &joint_q, array_t<float> &joint_qd, array_t<float> &mask, State &state);

// given maximal coordinate model computes ik (closest point projection)
void eval_ik(Model &model, State &state, array_t<float> &joint_q, array_t<float> &joint_qd);

}// namespace wp