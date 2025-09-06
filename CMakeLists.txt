cmake_minimum_required(VERSION 3.15)
project(TinyBitNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Catch2 3 REQUIRED)

# Gather sources
file(GLOB_RECURSE SOURCES src/*.cpp)
file(GLOB_RECURSE TEST_SOURCES tests/*.cpp)

# -----------------------------
# Library target for project code
# -----------------------------
add_library(tinybitlib ${SOURCES})
target_include_directories(tinybitlib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(tinybitlib
    PUBLIC
        Eigen3::Eigen
)

# -----------------------------
# Main executable
# -----------------------------
add_executable(tinybitnet src/main.cpp)
target_link_libraries(tinybitnet PRIVATE tinybitlib)

# -----------------------------
# Test executable
# -----------------------------
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE tinybitlib Catch2::Catch2WithMain)

