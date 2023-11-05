//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "context.h"

namespace wp {
Context::Context() { cuda_init(); }

int Context::cuda_init() {
    if (!check_cu(cuInit(0))) return -1;

    int deviceCount = 0;
    if (check_cu(cuDeviceGetCount(&deviceCount))) {
        g_devices.resize(deviceCount);

        for (int i = 0; i < deviceCount; i++) {
            CUdevice device;
            if (check_cu(cuDeviceGet(&device, i))) {
                // query device info
                g_devices[i].device = device;
                g_devices[i].ordinal = i;
                check_cu(cuDeviceGetName(g_devices[i].name, DeviceInfo::kNameLen, device));
                check_cu(cuDeviceGetAttribute(&g_devices[i].is_uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
                check_cu(cuDeviceGetAttribute(&g_devices[i].is_memory_pool_supported,
                                              CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device));
                int major = 0;
                int minor = 0;
                check_cu(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
                check_cu(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
                g_devices[i].arch = 10 * major + minor;

                g_device_map[device] = &g_devices[i];
            } else {
                return -1;
            }
        }
    } else {
        return -1;
    }

    return 0;
}

}  // namespace wp