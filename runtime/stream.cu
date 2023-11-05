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
Stream::Stream(Device &device) : device_{device} {
    ContextGuard guard(device.get_context(), true);
    check_cu(cuStreamCreate(static_cast<CUstream *>(stream), CU_STREAM_DEFAULT));
}

Stream::~Stream() {
    ContextGuard guard(device_.get_context(), true);
    check_cu(cuStreamDestroy(static_cast<CUstream>(stream)));
}

void Stream::record_event(Event &event) {
    ContextGuard guard(device_.get_context());

    check_cu(cuEventRecord(static_cast<CUevent>(event.handle()), static_cast<CUstream>(stream)));
}

void Stream::wait_event(Event &event) {
    ContextGuard guard(device_.get_context());

    check_cu(cuStreamWaitEvent(static_cast<CUstream>(stream), static_cast<CUevent>(event.handle()), 0));
}

void Stream::wait_stream(Stream &other_stream, Event &event) {
    ContextGuard guard(device_.get_context());

    check_cu(cuEventRecord(static_cast<CUevent>(event.handle()), static_cast<CUstream>(other_stream.stream)));
    check_cu(cuStreamWaitEvent(static_cast<CUstream>(stream), static_cast<CUevent>(event.handle()), 0));
}
}  // namespace wp
