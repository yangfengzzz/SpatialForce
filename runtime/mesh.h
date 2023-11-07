//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/mesh.h"

namespace wp {
class Mesh {
public:

private:
    uint64_t id_{};
    mesh_t descriptor_;
};
}  // namespace wp