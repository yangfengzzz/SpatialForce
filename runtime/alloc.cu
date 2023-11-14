//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/alloc.h"
#include "runtime/cuda_util.h"
#include "core/builtin.h"

namespace wp {
void *alloc_host(size_t s) {
    return malloc(s);
}
void *alloc_pinned(size_t s) {
    void *ptr;
    check_cuda(cudaMallocHost(&ptr, s));
    return ptr;
}
void *alloc_device(size_t s) {
    void *ptr;
    check_cuda(cudaMalloc(&ptr, s));
    return ptr;
}

void free_host(void *ptr) {
    free(ptr);
}
void free_pinned(void *ptr) {
    cudaFreeHost(ptr);
}
void free_device(void *ptr) {
    check_cuda(cudaFree(ptr));
}

// all memcpys are performed asynchronously
void memcpy_h2h(void *dest, void *src, size_t n) {
    memcpy(dest, src, n);
}
void memcpy_h2d(void *stream, void *dest, void *src, size_t n) {
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice, (cudaStream_t)stream));
}
void memcpy_d2h(void *stream, void *dest, void *src, size_t n) {
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
}
void memcpy_d2d(void *stream, void *dest, void *src, size_t n) {
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
}
void memcpy_peer(void *stream, void *dest, void *src, size_t n) {
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDefault, (cudaStream_t)stream));
}

__global__ void memset_kernel(int *dest, int value, size_t n) {
    const size_t tid = wp::grid_index();

    if (tid < n) {
        dest[tid] = value;
    }
}

// all memsets are performed asynchronously
void memset_host(void *dest, int value, size_t n) {
    if ((n % 4) > 0) {
        memset(dest, value, n);
    } else {
        const size_t num_words = n / 4;
        for (size_t i = 0; i < num_words; ++i)
            ((int *)dest)[i] = value;
    }
}

void memset_device(void *stream, void *dest, int value, size_t n) {
    if ((n % 4) > 0) {
        // for unaligned lengths fallback to CUDA memset
        check_cuda(cudaMemsetAsync(dest, value, n, (cudaStream_t)stream));
    } else {
        // custom kernel to support 4-byte values (and slightly lower host overhead)
        const size_t num_words = n / 4;
        wp_launch_device((cudaStream_t)stream, memset_kernel, num_words, ((int *)dest, value, num_words));
    }
}

// fill memory buffer with a value: this is a faster memtile variant
// for types bigger than one byte, but requires proper alignment of dst
template<typename T>
void memtile_value_host(T *dst, T value, size_t n) {
    while (n--)
        *dst++ = value;
}

// takes srcsize bytes starting at src and repeats them n times at dst (writes srcsize * n bytes in total):
void memtile_host(void *dst, const void *src, size_t srcsize, size_t n) {
    auto dst_addr = reinterpret_cast<size_t>(dst);
    auto src_addr = reinterpret_cast<size_t>(src);

    // try memtile_value first because it should be faster, but we need to ensure proper alignment
    if (srcsize == 8 && (dst_addr & 7) == 0 && (src_addr & 7) == 0)
        memtile_value_host(reinterpret_cast<int64_t *>(dst), *reinterpret_cast<const int64_t *>(src), n);
    else if (srcsize == 4 && (dst_addr & 3) == 0 && (src_addr & 3) == 0)
        memtile_value_host(reinterpret_cast<int32_t *>(dst), *reinterpret_cast<const int32_t *>(src), n);
    else if (srcsize == 2 && (dst_addr & 1) == 0 && (src_addr & 1) == 0)
        memtile_value_host(reinterpret_cast<int16_t *>(dst), *reinterpret_cast<const int16_t *>(src), n);
    else if (srcsize == 1)
        memset(dst, *reinterpret_cast<const int8_t *>(src), n);
    else {
        // generic version
        while (n--) {
            memcpy(dst, src, srcsize);
            dst = (int8_t *)dst + srcsize;
        }
    }
}

// fill memory buffer with a value: generic memtile kernel using memcpy for each element
__global__ void memtile_kernel(void *dst, const void *src, size_t srcsize, size_t n) {
    size_t tid = wp::grid_index();
    if (tid < n) {
        memcpy((int8_t *)dst + srcsize * tid, src, srcsize);
    }
}

// this should be faster than memtile_kernel, but requires proper alignment of dst
template<typename T>
__global__ void memtile_value_kernel(T *dst, T value, size_t n) {
    size_t tid = wp::grid_index();
    if (tid < n) {
        dst[tid] = value;
    }
}

void memtile_device(void *stream, void *dst, const void *src, size_t srcsize, size_t n) {
    auto dst_addr = reinterpret_cast<size_t>(dst);
    auto src_addr = reinterpret_cast<size_t>(src);

    // try memtile_value first because it should be faster, but we need to ensure proper alignment
    if (srcsize == 8 && (dst_addr & 7) == 0 && (src_addr & 7) == 0) {
        auto *p = reinterpret_cast<int64_t *>(dst);
        int64_t value = *reinterpret_cast<const int64_t *>(src);
        wp_launch_device((cudaStream_t)stream, memtile_value_kernel, n, (p, value, n));
    } else if (srcsize == 4 && (dst_addr & 3) == 0 && (src_addr & 3) == 0) {
        auto *p = reinterpret_cast<int32_t *>(dst);
        int32_t value = *reinterpret_cast<const int32_t *>(src);
        wp_launch_device((cudaStream_t)stream, memtile_value_kernel, n, (p, value, n));
    } else if (srcsize == 2 && (dst_addr & 1) == 0 && (src_addr & 1) == 0) {
        auto *p = reinterpret_cast<int16_t *>(dst);
        int16_t value = *reinterpret_cast<const int16_t *>(src);
        wp_launch_device((cudaStream_t)stream, memtile_value_kernel, n, (p, value, n));
    } else if (srcsize == 1) {
        check_cuda(cudaMemset(dst, *reinterpret_cast<const int8_t *>(src), n));
    } else {
        // generic version

        // TODO: use a persistent stream-local staging buffer to avoid allocs?
        void *src_device;
        check_cuda(cudaMalloc(&src_device, srcsize));
        check_cuda(cudaMemcpyAsync(src_device, src, srcsize, cudaMemcpyHostToDevice, (cudaStream_t)stream));

        wp_launch_device((cudaStream_t)stream, memtile_kernel, n, (dst, src_device, srcsize, n));

        check_cuda(cudaFree(src_device));
    }
}
}// namespace wp