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
