cmake_minimum_required(VERSION 3.10)

project(third_party LANGUAGES C CXX)

# googletest
add_subdirectory(googletest)

# google bechmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(benchmark EXCLUDE_FROM_ALL)