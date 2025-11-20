FetchContent_Declare(
  alpaka
  URL https://github.com/alpaka-group/alpaka/archive/refs/tags/2.0.0.tar.gz)
FetchContent_MakeAvailable(alpaka)
find_package(Boost 1.75.0 REQUIRED)

add_subdirectory(cpu)

if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
endif()

if(CMAKE_HIP_COMPILER)
  add_subdirectory(hip)
endif()
