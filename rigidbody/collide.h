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
CUDA_CALLABLE_DEVICE vec3f triangle_closest_point_barycentric(vec3f a, vec3f b, vec3f c, vec3f p);

void collide(Model& model, State& state, int edge_sdf_iter = 10);

}// namespace wp