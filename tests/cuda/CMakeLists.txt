enable_language(CUDA)

add_executable(association_map_cuda.out association_map.t.cu)
target_include_directories(
  association_map_cuda.out PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                   ${doctest_SOURCE_DIR}/doctest)
target_compile_options(association_map_cuda.out
                       PRIVATE --expt-relaxed-constexpr)
set_target_properties(association_map_cuda.out PROPERTIES CUDA_ARCHITECTURES
                                                          "native")
