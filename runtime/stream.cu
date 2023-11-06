//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_util.h"
#include "device.h"
#include "event.h"
#include "stream.h"

namespace wp {
Stream::Stream(Device& device) : device_{device} {
    CUstream stream;
    check_cu(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    handle_ = stream;
}

Stream::~Stream() {
    check_cu(cuStreamDestroy(static_cast<CUstream>(handle_)));
}

void Stream::record_event(Event& event) {
    check_cu(cuEventRecord(static_cast<CUevent>(event.handle()), static_cast<CUstream>(handle_)));
}

void Stream::wait_event(Event& event) {
    check_cu(cuStreamWaitEvent(static_cast<CUstream>(handle_), static_cast<CUevent>(event.handle()), 0));
}

void Stream::wait_stream(Stream& other_stream, Event& event) {
    check_cu(cuEventRecord(static_cast<CUevent>(event.handle()), static_cast<CUstream>(other_stream.handle_)));
    check_cu(cuStreamWaitEvent(static_cast<CUstream>(handle_), static_cast<CUevent>(event.handle()), 0));
}

void* Stream::handle() { return handle_; }

void Stream::memcpy_h2d(void* dest, void* src, size_t n) {
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice, (cudaStream_t)handle_));
}
void Stream::memcpy_d2h(void* dest, void* src, size_t n) {
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost, (cudaStream_t)handle_));
}
void Stream::memcpy_d2d(void* dest, void* src, size_t n) {
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToDevice, (cudaStream_t)handle_));
}

void Stream::memcpy_peer(void* dest, void* src, size_t n) {
    // NB: assumes devices involved support UVA
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDefault, (cudaStream_t)handle_));
}

void Stream::memset(void* dest, int value, size_t n) {
    check_cuda(cudaMemsetAsync(dest, value, n, (cudaStream_t)handle_));
}
}  // namespace wp
