//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "builtin.h"

namespace wp {
template<typename T, size_t N>
struct fixed_array_t {
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;
    using iterator = pointer;
    using const_iterator = const_pointer;

    CUDA_CALLABLE constexpr fixed_array_t();

    template<typename... Args>
    CUDA_CALLABLE constexpr fixed_array_t(const_reference first, Args... rest);

    CUDA_CALLABLE constexpr fixed_array_t(const fixed_array_t &other);

    CUDA_CALLABLE constexpr void fill(const_reference val);

    CUDA_CALLABLE constexpr reference operator[](size_t i);

    CUDA_CALLABLE constexpr const_reference operator[](size_t i) const;

    CUDA_CALLABLE constexpr bool operator==(const fixed_array_t &other) const;

    CUDA_CALLABLE constexpr bool operator!=(const fixed_array_t &other) const;

    template<typename... Args>
    CUDA_CALLABLE constexpr void set_at(size_t i, const_reference first, Args... rest);

    template<typename... Args>
    CUDA_CALLABLE constexpr void set_at(size_t i, const_reference first);

    T data[N];
};

template<typename T, size_t N>
CUDA_CALLABLE constexpr fixed_array_t<T, N>::fixed_array_t() { fill(T{}); }

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE constexpr fixed_array_t<T, N>::fixed_array_t(const_reference first, Args... rest) {
    static_assert(sizeof...(Args) == N - 1, "Number of arguments should be equal to the size of the vector.");
    set_at(0, first, rest...);
}

template<typename T, size_t N>
CUDA_CALLABLE constexpr fixed_array_t<T, N>::fixed_array_t(const fixed_array_t &other) {
    for (size_t i = 0; i < N; ++i) {
        data[i] = other[i];
    }
}

template<typename T, size_t N>
CUDA_CALLABLE constexpr void fixed_array_t<T, N>::fill(const_reference val) {
    for (size_t i = 0; i < N; ++i) {
        data[i] = val;
    }
}

template<typename T, size_t N>
CUDA_CALLABLE constexpr typename fixed_array_t<T, N>::reference fixed_array_t<T, N>::operator[](size_t i) {
    return data[i];
}

template<typename T, size_t N>
CUDA_CALLABLE constexpr typename fixed_array_t<T, N>::const_reference fixed_array_t<T, N>::operator[](size_t i) const {
    return data[i];
}

template<typename T, size_t N>
CUDA_CALLABLE constexpr bool fixed_array_t<T, N>::operator==(const fixed_array_t &other) const {
    for (size_t i = 0; i < N; ++i) {
        if (data[i] != other._elements[i]) {
            return false;
        }
    }

    return true;
}

template<typename T, size_t N>
CUDA_CALLABLE constexpr bool fixed_array_t<T, N>::operator!=(const fixed_array_t &other) const {
    return *this != other;
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE constexpr void fixed_array_t<T, N>::set_at(size_t i, const_reference first, Args... rest) {
    data[i] = first;
    set_at(i + 1, rest...);
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE constexpr void fixed_array_t<T, N>::set_at(size_t i, const_reference first) {
    data[i] = first;
}

}// namespace wp