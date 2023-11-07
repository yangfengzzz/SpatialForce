//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/cuda_util.h"
#include "runtime/device.h"
#include "runtime/scan.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/device/device_scan.cuh>

namespace wp {

template <typename T>
void scan(Stream& stream, const T* values_in, T* values_out, int n, bool inclusive) {
    auto s = static_cast<cudaStream_t>(stream.handle());

    // compute temporary memory required
    size_t scan_temp_size;
    if (inclusive) {
        check_cuda(cub::DeviceScan::InclusiveSum(nullptr, scan_temp_size, values_in, values_out, n));
    } else {
        check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_size, values_in, values_out, n));
    }

    void* temp_buffer = Device::alloc(scan_temp_size);

    // scan
    if (inclusive) {
        check_cuda(cub::DeviceScan::InclusiveSum(temp_buffer, scan_temp_size, values_in, values_out, n, s));
    } else {
        check_cuda(cub::DeviceScan::ExclusiveSum(temp_buffer, scan_temp_size, values_in, values_out, n, s));
    }

    Device::free(temp_buffer);
}

template void scan(Stream& stream, const int*, int*, int, bool);
template void scan(Stream& stream, const float*, float*, int, bool);
}  // namespace wp