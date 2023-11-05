//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace wp {
class Device;

class Allocator {
public:
    explicit Allocator(Device& device);

    void* alloc(size_t s);

    void free(void* ptr);

private:
    Device& device_;
};
}  // namespace wp