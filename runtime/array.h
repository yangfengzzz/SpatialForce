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
    explicit Array(shape_t shape);

    ~Array();

    const array_t<T>& handle() const { return handle_; }

    array_t<T>& handle() { return handle_; }

private:
    array_t<T> handle_;
};

template <typename T>
Array<T>::Array(shape_t shape) {
    auto capacity = shape.size() * sizeof(T);
    auto ptr = Device::alloc(capacity);
    switch (shape.dim()) {
        case 1:
            handle_ = array_t<T>(ptr, shape.dims[0]);
        case 2:
            handle_ = array_t<T>(ptr, shape.dims[0], shape.dims[1]);
        case 3:
            handle_ = array_t<T>(ptr, shape.dims[0], shape.dims[1], shape.dims[2]);
        case 4:
            handle_ = array_t<T>(ptr, shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]);
    }
}

template <typename T>
Array<T>::~Array() {
    Device::free(handle_.data);
}

}  // namespace wp