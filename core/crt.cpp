//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "crt.h"

#include <cassert>
#include <cmath>
#include <cstdio>

extern "C" WP_API int _wp_isfinite(double x) { return std::isfinite(x); }

extern "C" WP_API void _wp_assert(const char* expression, const char* file, unsigned int line) {
    fflush(stdout);
    fprintf(stderr,
            "Assertion failed: '%s'\n"
            "At '%s:%d'\n",
            expression, file, line);
    fflush(stderr);

    // Now invoke the standard assert(), which may abort the program or break
    // into the debugger as decided by the runtime environment.
    assert(false && "assert() failed");
}