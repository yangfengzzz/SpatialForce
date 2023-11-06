//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "allocator.h"
#include "stream.h"

namespace wp {
class Device {
public:
    Device();

    void* context();

    Allocator& allocator();

    Stream& stream();

public:
    void memcpy_h2d(void* dest, void* src, size_t n);
    void memcpy_d2h(void* dest, void* src, size_t n);
    void memcpy_d2d(void* dest, void* src, size_t n);
    void memcpy_peer(void* dest, void* src, size_t n);

private:
    void* context_;
    Allocator allocator_;
    Stream stream_;
};
}  // namespace wp