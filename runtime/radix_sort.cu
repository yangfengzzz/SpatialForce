//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <cub/cub.cuh>

#include "cuda_util.h"
#include "device.h"
#include "radix_sort.h"

namespace wp {
std::map<void*, RadixSort::RadixSortTemp> RadixSort::g_radix_sort_temp_map;

RadixSort::RadixSort(Stream& stream) : stream_{stream} {}

void RadixSort::reserve(int n, void** mem_out, size_t* size_out) {
    cub::DoubleBuffer<int> d_keys;
    cub::DoubleBuffer<int> d_values;

    // compute temporary memory required
    size_t sort_temp_size;
    check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, sort_temp_size, d_keys, d_values, n, 0, 32,
                                               (cudaStream_t)stream_.handle()));

    RadixSortTemp& temp = g_radix_sort_temp_map[stream_.device().context()];

    if (sort_temp_size > temp.size) {
        Device::free(temp.mem);
        temp.mem = Device::alloc(sort_temp_size);
        temp.size = sort_temp_size;
    }

    if (mem_out) *mem_out = temp.mem;
    if (size_out) *size_out = temp.size;
}

void RadixSort::sort_pairs(int* keys, int* values, int n) {
    cub::DoubleBuffer<int> d_keys(keys, keys + n);
    cub::DoubleBuffer<int> d_values(values, values + n);

    RadixSortTemp temp;
    reserve(n, &temp.mem, &temp.size);

    // sort
    check_cuda(cub::DeviceRadixSort::SortPairs(temp.mem, temp.size, d_keys, d_values, n, 0, 32,
                                               (cudaStream_t)stream_.handle()));

    if (d_keys.Current() != keys) stream_.memcpy_d2d(keys, d_keys.Current(), sizeof(int) * n);

    if (d_values.Current() != values) stream_.memcpy_d2d(values, d_values.Current(), sizeof(int) * n);
}
}  // namespace wp