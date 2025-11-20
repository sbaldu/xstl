enable_language(CUDA)

set_source_files_properties(../binary_associator.cpp PROPERTIES LANGUAGE CUDA)
add_executable(binary_associator_alpaka_cuda.out ../binary_associator.cpp)
target_include_directories(binary_associator_alpaka_cuda.out
                           PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_link_libraries(binary_associator_alpaka_cuda.out
                      PRIVATE alpaka::alpaka Boost::boost benchmark::benchmark)
target_compile_definitions(binary_associator_alpaka_cuda.out
                           PRIVATE -DALPAKA_ACC_GPU_CUDA_ENABLED)
target_compile_options(binary_associator_alpaka_cuda.out
                       PRIVATE --expt-relaxed-constexpr)
set_target_properties(binary_associator_alpaka_cuda.out
                      PROPERTIES CUDA_ARCHITECTURES "native")
