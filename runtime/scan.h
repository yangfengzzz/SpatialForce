//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace wp {
class Stream;

template <typename T>
void scan(Stream& stream, const T* values_in, T* values_out, int n, bool inclusive = true);
}  // namespace wp