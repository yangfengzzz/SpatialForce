//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#if !defined(__CUDA_ARCH__)
#if defined(_WIN32)
#define WP_API __declspec(dllexport)
#else
#define WP_API __attribute__((visibility("default")))
#endif
#else
#define WP_API
#endif

#if !defined(__CUDA_ARCH__)

// Helper for implementing assert() macro
extern "C" WP_API void _wp_assert(const char* message, const char* file, unsigned int line);

// Helper for implementing isfinite()
extern "C" WP_API int _wp_isfinite(double);

#endif  // !__CUDA_ARCH__

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
