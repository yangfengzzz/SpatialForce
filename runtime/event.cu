//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_util.h"
#include "device.h"
#include "event.h"

namespace wp {
Event::Event(bool enable_timing) {
    int flags = CU_EVENT_DEFAULT;
    if (!enable_timing) {
        flags |= CU_EVENT_DISABLE_TIMING;
    }

    check_cu(cuEventCreate(reinterpret_cast<CUevent *>(&event_), flags));
}

Event::~Event() { check_cu(cuEventDestroy(static_cast<CUevent>(event_))); }

}  // namespace wp