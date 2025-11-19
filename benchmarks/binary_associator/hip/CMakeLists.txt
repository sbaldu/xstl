enable_language(HIP)

add_executable(binary_associator_hip.out binary_associator.cpp)
target_include_directories(binary_associator_hip.out
                           PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_link_libraries(binary_associator_hip.out PRIVATE benchmark::benchmark)
