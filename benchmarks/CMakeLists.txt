cmake_minimum_required(VERSION 3.16.0)

project(XSTL_benchmarks LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS
    " -O2 -funroll-loops -funsafe-math-optimizations -ftree-vectorize -march=native"
)

find_package(benchmark REQUIRED)

add_subdirectory(binary_associator)
