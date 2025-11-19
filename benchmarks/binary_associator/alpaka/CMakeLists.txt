find_package(alpaka)
if(NOT alpaka_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    alpaka
    URL https://github.com/alpaka-group/alpaka/archive/refs/tags/2.0.0.tar.gz)
  FetchContent_MakeAvailable(alpaka)
endif()

add_executable(binary_associator_alpaka.out binary_associator.cpp)
target_include_directories(binary_associator_alpaka.out
                           PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_link_libraries(binary_associator_alpaka.out PRIVATE alpaka::alpaka
                                                           benchmark::benchmark)
target_compile_definitions(binary_associator_alpaka.out
                           PRIVATE -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
