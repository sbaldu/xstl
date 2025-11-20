enable_language(HIP)

set_source_files_properties(../binary_associator.cpp PROPERTIES LANGUAGE HIP)
add_executable(binary_associator_alpaka_hip.out ../binary_associator.cpp)
target_include_directories(binary_associator_alpaka_hip.out
                           PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_link_libraries(binary_associator_alpaka_hip.out
                      PRIVATE alpaka::alpaka Boost::boost benchmark::benchmark)
target_compile_definitions(binary_associator_alpaka_hip.out
                           PRIVATE -DALPAKA_ACC_GPU_HIP_ENABLED)
