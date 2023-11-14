//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

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
}// namespace wp