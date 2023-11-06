//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/context.h"
#include "stream.h"

namespace wp {
class Device {
public:
    explicit Device(Context::DeviceInfo info);

    void* context();

    Stream create_stream();

    static void* alloc(size_t s);

    static void free(void* ptr);

private:
    Context::DeviceInfo info_;
    void* context_;
};
}  // namespace wp