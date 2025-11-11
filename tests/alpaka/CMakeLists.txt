FetchContent_Declare(
  alpaka
  URL https://github.com/alpaka-group/alpaka/archive/refs/tags/2.0.0.tar.gz)
FetchContent_MakeAvailable(alpaka)

add_executable(association_map_alpaka_cpu.out association_map.t.cpp)
target_include_directories(
  association_map_alpaka_cpu.out
  PRIVATE ${CMAKE_SOURCE_DIR}/../include ${doctest_SOURCE_DIR}/doctest
          ${alpaka_SOURCE_DIR}/include)
target_compile_definitions(association_map_alpaka_cpu.out
                           PRIVATE -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
