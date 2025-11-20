add_executable(binary_associator_alpaka_cpu.out ../binary_associator.cpp)
target_include_directories(binary_associator_alpaka_cpu.out
                           PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_link_libraries(binary_associator_alpaka_cpu.out
                      PRIVATE alpaka::alpaka Boost::boost benchmark::benchmark)
target_compile_definitions(binary_associator_alpaka_cpu.out
                           PRIVATE -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
