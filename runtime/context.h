//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <map>
#include <vector>

#include "cuda_util.h"

namespace wp {
class Device;

class Context {
public:
    struct DeviceInfo {
        static constexpr int kNameLen = 128;

        CUdevice device = -1;
        int ordinal = -1;
        char name[kNameLen] = "";
        int arch = 0;
        int is_uva = 0;
        int is_memory_pool_supported = 0;
    };

    Context();

    Device creat_device();

private:
    // cached info for all devices, indexed by ordinal
    std::vector<DeviceInfo> g_devices;

    // maps CUdevice to DeviceInfo
    std::map<CUdevice, DeviceInfo *> g_device_map;

    int active_index{};
};
}  // namespace wp