//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace wp {
class Device;

class Event {
public:
    explicit Event(Device &device, bool enable_timing = false);

    ~Event();

    void *handle() { return event_; }

private:
    Device &device_;
    void *event_{};
};
}  // namespace wp