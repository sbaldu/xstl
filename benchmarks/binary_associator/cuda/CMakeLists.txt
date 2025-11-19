enable_language(CUDA)

add_executable(binary_associator_cuda.out binary_associator.cu)
target_include_directories(binary_associator_cuda.out
                           PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_link_libraries(binary_associator_cuda.out PRIVATE benchmark::benchmark)
target_compile_options(binary_associator_cuda.out
                       PRIVATE --expt-relaxed-constexpr)
set_target_properties(binary_associator_cuda.out PROPERTIES CUDA_ARCHITECTURES
                                                            "native")
