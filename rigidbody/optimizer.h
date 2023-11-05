//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <functional>
#include <vector>

#include "core/array.h"
#include "core/mat.h"
#include "model.h"

namespace wp {
class Optimizer {
public:
    Optimizer();

    void solve(array_t<float> x, const std::function<void()> &grad_func,
               int max_iters = 20, float alpha = 0.01, bool report = false);
private:
};
}// namespace wp