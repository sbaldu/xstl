find_package(alpaka)
if(NOT alpaka_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    alpaka
    URL https://github.com/alpaka-group/alpaka/archive/refs/tags/2.0.0.tar.gz)
  FetchContent_MakeAvailable(alpaka)
endif()
find_package(Boost 1.75.0 REQUIRED)

add_subdirectory(cpu)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
  add_subdirectory(hip)
endif()
