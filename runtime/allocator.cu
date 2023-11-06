//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "allocator.h"
#include "cuda_util.h"
#include "device.h"

namespace wp {
Allocator::Allocator(Device &device) : device_{device} {}

void *Allocator::alloc(size_t s) {
    ContextGuard guard(device_.context());

    void *ptr;
    check_cuda(cudaMalloc(&ptr, s));
    return ptr;
}

void Allocator::free(void *ptr) {
    ContextGuard guard(device_.context());
    check_cuda(cudaFree(ptr));
}

}  // namespace wp
