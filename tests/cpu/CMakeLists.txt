
add_executable(association_map_cpu.out association_map.t.cpp)
target_include_directories(
	association_map_cpu.out
	PRIVATE ${CMAKE_SOURCE_DIR}/../include ${doctest_SOURCE_DIR}/doctest)
