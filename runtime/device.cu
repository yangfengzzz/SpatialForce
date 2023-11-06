//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_util.h"
#include "device.h"

namespace wp {
Device::Device(Context::DeviceInfo info) : info_(info) {
    CUcontext context = nullptr;
    check_cu(cuDevicePrimaryCtxRetain(&context, info.device));
    context_ = context;
}

void *Device::context() { return context_; }

void *Device::alloc(size_t s) {
    void *ptr;
    check_cuda(cudaMalloc(&ptr, s));
    return ptr;
}

void Device::free(void *ptr) { check_cuda(cudaFree(ptr)); }

Stream Device::create_stream() { return Stream(*this); }

}  // namespace wp
