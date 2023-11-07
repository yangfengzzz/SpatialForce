//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <map>

namespace wp {
class Stream;

class RadixSort {
public:
    explicit RadixSort(Stream& stream);

    void reserve(int n, void** mem_out = nullptr, size_t* size_out = nullptr);

    void sort_pairs(int* keys, int* values, int n);

private:
    Stream& stream_;

    // temporary buffer for radix sort
    struct RadixSortTemp {
        void* mem = nullptr;
        size_t size = 0;
    };

    // map temp buffers to CUDA contexts
    static std::map<void*, RadixSortTemp> g_radix_sort_temp_map;
};
}  // namespace wp