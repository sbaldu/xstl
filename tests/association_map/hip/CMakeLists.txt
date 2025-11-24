enable_language(HIP)

add_executable(association_map_hip.out association_map.t.cpp)
add_test(NAME association_map_hip.out COMMAND association_map_hip.out)
target_include_directories(
  association_map_hip.out PRIVATE ${CMAKE_SOURCE_DIR}/../include
                                  ${doctest_SOURCE_DIR}/doctest)
