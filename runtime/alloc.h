//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/array.h"
#include <vector>

namespace wp {
void *alloc_host(size_t s);
void *alloc_pinned(size_t s);
void *alloc_device(size_t s);

void free_host(void *ptr);
void free_pinned(void *ptr);
void free_device(void *ptr);

// all memcpys are performed asynchronously
void memcpy_h2h(void *dest, void *src, size_t n);
void memcpy_h2d(void *stream, void *dest, void *src, size_t n);
void memcpy_d2h(void *stream, void *dest, void *src, size_t n);
void memcpy_d2d(void *stream, void *dest, void *src, size_t n);
void memcpy_peer(void *stream, void *dest, void *src, size_t n);

// all memsets are performed asynchronously
void memset_host(void *dest, int value, size_t n);
void memset_device(void *stream, void *dest, int value, size_t n);

// takes srcsize bytes starting at src and repeats them n times at dst (writes srcsize * n bytes in total):
void memtile_host(void *dest, const void *src, size_t srcsize, size_t n);
void memtile_device(void *stream, void *dest, const void *src, size_t srcsize, size_t n);

template<typename T>
array_t<T> alloc_array(const std::vector<T> &src) {
    auto count = sizeof(T) * src.size();
    auto d = alloc_device(count);
    memcpy_h2d(nullptr, d, (void*)src.data(), count);
    return {(T*)d, (int)src.size()};
}

template<typename T>
void free_array(array_t<T> array) {
    free_device(array.data);
    array.data = nullptr;
}

template<typename T>
void copy_array_d2h(array_t<T> array, std::vector<T> &src) {
    auto count = sizeof(T) * src.size();
    memcpy_d2h(nullptr, src.data(), array.data, count);
}

}// namespace wp