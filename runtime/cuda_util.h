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
#if _DEBUG
// helper for launching kernels (synchronize + error checking after each kernel)
#define wp_launch_device(context, kernel, dim, args)                                      \
    {                                                                                     \
        if (dim) {                                                                        \
            ContextGuard guard(context);                                                  \
            const int num_threads = 256;                                                  \
            const int num_blocks = (dim + num_threads - 1) / num_threads;                 \
            kernel<<<num_blocks, 256, 0, (cudaStream_t)cuda_stream_get_current()>>> args; \
            check_cuda(cuda_context_check(WP_CURRENT_CONTEXT));                           \
        }                                                                                 \
    }
#else
// helper for launching kernels (no error checking)
#define wp_launch_device(context, kernel, dim, args)                                      \
    {                                                                                     \
        if (dim) {                                                                        \
            ContextGuard guard(context);                                                  \
            const int num_threads = 256;                                                  \
            const int num_blocks = (dim + num_threads - 1) / num_threads;                 \
            kernel<<<num_blocks, 256, 0, (cudaStream_t)cuda_stream_get_current()>>> args; \
        }                                                                                 \
    }
#endif  // _DEBUG
#endif  // defined(__CUDACC__)

bool init_cuda_driver();

bool check_cuda_result(cudaError_t code, const char *file, int line);
inline bool check_cuda_result(uint64_t code, const char *file, int line) {
    return check_cuda_result(static_cast<cudaError_t>(code), file, line);
}

bool check_cu_result(CUresult result, const char *file, int line);

//
// Scoped CUDA context guard
//
// Behaviour on entry
// - If the given `context` is NULL, do nothing.
// - If the given `context` is the same as the current context, do nothing.
// - If the given `context` is different from the current context, make the given context current.
//
// Behaviour on exit
// - If the current context did not change on entry, do nothing.
// - If the `restore` flag was true on entry, make the previous context current.
//
// Default exit behaviour policy
// - If the `restore` flag is omitted on entry, fall back on the global `always_restore` flag.
// - This allows us to easily change the default behaviour of the guards.
//
class ContextGuard {
public:
    // default policy for restoring contexts
    static bool always_restore;

    explicit ContextGuard(CUcontext context, bool restore = always_restore) : need_restore(false) {
        if (context) {
            if (check_cu(cuCtxGetCurrent(&prev_context)) && context != prev_context)
                need_restore = check_cu(cuCtxSetCurrent(context)) && restore;
        }
    }

    explicit ContextGuard(void *context, bool restore = always_restore)
        : ContextGuard(static_cast<CUcontext>(context), restore) {}

    ~ContextGuard() {
        if (need_restore) check_cu(cuCtxSetCurrent(prev_context));
    }

private:
    CUcontext prev_context{};
    bool need_restore;
};

// Pass this value to device functions as the `context` parameter to bypass unnecessary context management.
// This works in conjuntion with ContextGuards, which do nothing if the given context is NULL.
// Using this variable instead of passing NULL directly aids readability and makes the intent clear.
constexpr void *WP_CURRENT_CONTEXT = nullptr;

}  // namespace wp