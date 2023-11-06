//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_util.h"
#include "device.h"

namespace wp {
Device::Device() : allocator_(*this), stream_(*this) {}

void* Device::context() { return context_; }

Allocator& Device::allocator() { return allocator_; }

Stream& Device::stream() { return stream_; }

void Device::memcpy_h2d(void* dest, void* src, size_t n) {
    ContextGuard guard(context_);

    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice, (cudaStream_t)stream().handle()));
}
void Device::memcpy_d2h(void* dest, void* src, size_t n) {
    ContextGuard guard(context_);

    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost, (cudaStream_t)stream().handle()));
}
void Device::memcpy_d2d(void* dest, void* src, size_t n) {
    ContextGuard guard(context_);

    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToDevice, (cudaStream_t)stream().handle()));
}

void Device::memcpy_peer(void* dest, void* src, size_t n) {
    ContextGuard guard(context_);

    // NB: assumes devices involved support UVA
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDefault, (cudaStream_t)stream().handle()));
}

}  // namespace wp
