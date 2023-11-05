//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/array.h"
#include "device.h"

namespace wp {
template <typename T>
class Array {
public:
    Array(Device& device, shape_t shape, uint32_t strides);

    array_t<T> handle() { return handle_; }

private:
    Device& device_;
    array_t<T> handle_;
};

template <typename T>
Array<T>::Array(Device& device, shape_t shape, uint32_t strides) : device_{device} {}

}  // namespace wp