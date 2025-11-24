enable_language(HIP)

set_source_files_properties(../association_map.t.cpp PROPERTIES LANGUAGE HIP)
add_executable(association_map_alpaka_hip.out ../association_map.t.cpp)
add_test(NAME association_map_alpaka_hip.out
         COMMAND association_map_alpaka_hip.out)
target_include_directories(
  association_map_alpaka_hip.out PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                         ${doctest_SOURCE_DIR}/doctest)
target_link_libraries(association_map_alpaka_hip.out PRIVATE alpaka::alpaka
                                                             Boost::boost)
target_compile_definitions(association_map_alpaka_hip.out
                           PRIVATE -DALPAKA_ACC_GPU_HIP_ENABLED)
