#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.16)

project(fields LANGUAGES C CXX CUDA)

set(COMMON_FILES
        geometry.h
        geometry_trait.h
        template_geometry.h
        templates/interval.h
        templates/triangle.h
        templates/tetrahedron.h
        mesh.h
        tmp.cu
)

set(HOST_FILES
        host/grid_host.h
        host/grid_host.cu
        host/geometry_host.h
        host/geometry_host.cu
        host/mesh_host.h
        host/mesh_host.cu
        host/poly_info_1d_host.h
        host/poly_info_1d_host.cu
        host/poly_info_2d_host.h
        host/poly_info_2d_host.cu
        host/poly_info_3d_host.h
        host/poly_info_3d_host.cu
        host/recon_auxiliary_host.h
        host/recon_auxiliary_host.cu
        host/grid_data_host.h
        host/grid_data_host.cu
        host/grid_system_data_host.h
        host/grid_system_data_host.cu
)

set(IO_FILES
        io/simplest_mesh.h
        io/simplest_mesh.cpp
        io/gmsh2d_io.h
        io/gmsh2d_io.cpp
)

source_group("common\\" FILES ${COMMON_FILES})

set(PROJECT_FILES
        ${COMMON_FILES}
        ${HOST_FILES}
        ${IO_FILES}
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