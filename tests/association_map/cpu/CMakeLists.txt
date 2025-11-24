add_executable(association_map_cpu.out association_map.t.cpp)
add_test(NAME association_map_cpu_test COMMAND association_map_cpu.out)
target_include_directories(
  association_map_cpu.out PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                  ${doctest_SOURCE_DIR}/doctest)
if(COVERAGE)
  target_compile_options(association_map_cpu.out PRIVATE -g --coverage)
  target_link_options(association_map_cpu.out PRIVATE --coverage)
endif()
