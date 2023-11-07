//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/mesh.h"
#include "runtime/array.h"

namespace wp {
class Mesh {
public:
    Mesh(Stream& stream, const Array<vec3>& points, const Array<vec3>& velocities, const Array<int>& indices, bool support_winding_number);

    ~Mesh();

    void refit();

private:
    void bvh_refit_with_solid_angle_device(bvh_t& bvh, mesh_t& mesh);

    Stream& stream_;
    uint64_t id_{};
    mesh_t descriptor_;
};
}  // namespace wp