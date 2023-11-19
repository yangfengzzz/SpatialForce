#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

# create sample app project
project(sim-tests LANGUAGES C CXX CUDA)

set(SRC
        main.cpp
        volume_integrator_tests.cu
)

add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME} PRIVATE fields runtime gtest gtest_main gmock gmock_main CUDA::cuda_driver)

