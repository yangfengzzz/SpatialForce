//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>

#include <cstdio>

namespace wp {
#define check_cuda(code) (check_cuda_result(code, __FILE__, __LINE__))
#define check_cu(code) (check_cu_result(code, __FILE__, __LINE__))

#if defined(__CUDACC__)
// helper for launching kernels (no error checking)
#define wp_launch_device(stream, kernel, dim, args)                       \
    {                                                                     \
        if (dim) {                                                        \
            const int num_threads = 256;                                  \
            const int num_blocks = (dim + num_threads - 1) / num_threads; \
            kernel<<<num_blocks, 256, 0, stream>>> args;                  \
        }                                                                 \
    }
#endif  // defined(__CUDACC__)

bool check_cuda_result(cudaError_t code, const char *file, int line);

inline bool check_cuda_result(uint64_t code, const char *file, int line) {
    return check_cuda_result(static_cast<cudaError_t>(code), file, line);
}

bool check_cu_result(CUresult result, const char *file, int line);

}  // namespace wp