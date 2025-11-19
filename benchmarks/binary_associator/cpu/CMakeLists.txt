add_executable(binary_associator_cpu.out binary_associator.cpp)
target_include_directories(binary_associator_cpu.out
                           PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_link_libraries(binary_associator_cpu.out PRIVATE benchmark::benchmark)
