//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/spatial.h"

namespace wp {
constexpr float PI = 3.14159265359f;
constexpr float PI_2 = 1.57079632679f;

CUDA_CALLABLE vec3f velocity_at_point(spatial_vectorf qd, vec3f r);

CUDA_CALLABLE quatf quat_twist(vec3f axis, quatf q);

CUDA_CALLABLE float quat_twist_angle(vec3f axis, quatf q);

CUDA_CALLABLE vec3 quat_decompose(quatf q);

CUDA_CALLABLE vec3 quat_to_rpy(quatf q);

CUDA_CALLABLE vec3f quat_to_euler(quatf q, int i, int j, int k);

CUDA_CALLABLE quatf quat_between_vectors(vec3f a, vec3f b);

CUDA_CALLABLE spatial_vector transform_twist(transformf t, spatial_vectorf x);

CUDA_CALLABLE spatial_vector transform_wrench(transformf t, spatial_vectorf x);

CUDA_CALLABLE vec3f vec_min(vec3f a, vec3f b);

CUDA_CALLABLE vec3f vec_max(vec3f a, vec3f b);

CUDA_CALLABLE vec3f vec_abs(vec3f a);
}// namespace wp