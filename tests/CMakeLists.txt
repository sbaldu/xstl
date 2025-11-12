cmake_minimum_required(VERSION 3.16.0)

project(XSTL_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

string(
  APPEND
  CMAKE_CXX_FLAGS_DEBUG
  " -DCLUE_DEBUG -D_GLIBCXX_ASSERTIONS -O0 -Wall -Wextra -Wpedantic -Wshadow -Wimplicit-fallthrough -Wextra-semi -Wold-style-cast -g -pg -fsanitize=address"
)
set(CMAKE_CXX_FLAGS_RELEASE
    " -O2 -funroll-loops -funsafe-math-optimizations -ftree-vectorize -march=native"
)

include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG dev)
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
  FetchContent_MakeAvailable(doctest)
endif()

add_subdirectory(cpu)
add_subdirectory(alpaka)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
  add_subdirectory(hip)
endif()
