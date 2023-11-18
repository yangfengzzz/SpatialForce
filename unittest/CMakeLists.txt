#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

# create sample app project
project(sim-tests LANGUAGES C CXX)

set(SRC
        main.cpp
)

add_executable(${PROJECT_NAME} WIN32 ${SRC})

target_link_libraries(${PROJECT_NAME} PRIVATE fields gtest gtest_main gmock gmock_main)

