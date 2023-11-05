//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/array.h"
#include "core/mat.h"
#include <vector>

namespace wp {
void compute_sphere_inertia(float density, float r);

void compute_capsule_inertia(float density, float r, float h);

void compute_cylinder_inertia(float density, float r, float h);

void compute_cone_inertia(float density, float r, float h);

void compute_box_inertia(float density, float w, float h, float d);

void compute_mesh_inertia(float density, const std::vector<float>& vertices,
                          const std::vector<float>& indices, bool is_solid = true,
                          const std::vector<float>& thickness = {0.001});

void transform_inertia(float m, float I, float p, float q);

}// namespace wp