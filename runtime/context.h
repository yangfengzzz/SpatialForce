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
class Context {
public:
    Context();

    int cuda_init();

private:
    struct DeviceInfo {
        static constexpr int kNameLen = 128;

        CUdevice device = -1;
        int ordinal = -1;
        char name[kNameLen] = "";
        int arch = 0;
        int is_uva = 0;
        int is_memory_pool_supported = 0;
    };

    struct ContextInfo {
        DeviceInfo *device_info = nullptr;

        CUstream stream = nullptr;  // created when needed
    };

    // cached info for all devices, indexed by ordinal
    std::vector<DeviceInfo> g_devices;

    // maps CUdevice to DeviceInfo
    std::map<CUdevice, DeviceInfo *> g_device_map;

    // cached info for all known contexts
    std::map<CUcontext, ContextInfo> g_contexts;
};
}  // namespace wp