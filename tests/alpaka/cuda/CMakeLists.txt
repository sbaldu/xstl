enable_language(CUDA)

set_source_files_properties(../association_map.t.cpp PROPERTIES LANGUAGE CUDA)
add_executable(association_map_alpaka_cuda.out ../association_map.t.cpp)
target_include_directories(
  association_map_alpaka_cuda.out PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                          ${doctest_SOURCE_DIR}/doctest)
target_link_libraries(association_map_alpaka_cuda.out PRIVATE alpaka::alpaka
                                                              Boost::boost)
target_compile_definitions(association_map_alpaka_cuda.out
                           PRIVATE -DALPAKA_ACC_GPU_CUDA_ENABLED)
target_compile_options(association_map_alpaka_cuda.out
                       PRIVATE --expt-relaxed-constexpr)
set_target_properties(association_map_alpaka_cuda.out
                      PROPERTIES CUDA_ARCHITECTURES "native")
