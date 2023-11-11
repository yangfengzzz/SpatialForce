#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.16)

project(core LANGUAGES C CXX CUDA)

set(COMMON_FILES
        crt.h
        crt.cpp
        builtin.h
        initializer_array.h
        vec.h
        mat.h
        quat.h
        array.h
        fixed_array.h
        fabric.h
        spatial.h
        rand.h
        noise.h
        range.h
        solid_angle.h
        intersect.h
        intersect_tri.h
        svd.h
        matnn.h
        #
        hashgrid.h
        bounds.h
        bvh.h
        mesh.h
)

source_group("common\\" FILES ${COMMON_FILES})

set(PROJECT_FILES
        ${COMMON_FILES}
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