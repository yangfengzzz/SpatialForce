#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.16)

project(runtime LANGUAGES C CXX CUDA)

set(COMMON_FILES
        cuda_util.h
        cuda_util.cu
        device.h
        device.cu
        context.h
        context.cu
        graph.h
        graph.cu
        stream.h
        stream.cu
        event.h
        event.cu
        alloc.h
        alloc.cu
)

set(TYPE_FILES
        array.h
        radix_sort.h
        radix_sort.cu
        scan.h
        scan.cu
        hash_grid.h
        hash_grid.cu
        bvh.h
        bvh.cu
        mesh.h
        mesh.cu
)

source_group("common\\" FILES ${COMMON_FILES})

set(PROJECT_FILES
        ${COMMON_FILES}
        ${TYPE_FILES}
)

add_library(${PROJECT_NAME} OBJECT ${PROJECT_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} .
        ../)

if (${NEED_LINK_ATOMIC})
    target_link_libraries(${PROJECT_NAME} PUBLIC atomic)
endif ()

if (CMAKE_DO_CLANG_TIDY)
    set_target_properties(framework PROPERTIES CXX_CLANG_TIDY "${CMAKE_DO_CLANG_TIDY}")
endif ()