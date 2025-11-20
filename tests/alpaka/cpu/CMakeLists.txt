add_executable(association_map_alpaka_cpu.out ../association_map.t.cpp)
target_include_directories(
  association_map_alpaka_cpu.out PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                         ${doctest_SOURCE_DIR}/doctest)
target_link_libraries(association_map_alpaka_cpu.out PRIVATE alpaka::alpaka
                                                             Boost::boost)
target_compile_definitions(association_map_alpaka_cpu.out
                           PRIVATE -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
if(COVERAGE)
  target_compile_options(association_map_alpaka_cpu.out PRIVATE -g --coverage)
  target_link_options(association_map_alpaka_cpu.out PRIVATE --coverage)
endif()
