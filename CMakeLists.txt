cmake_minimum_required(VERSION 3.16.0)

project(
  xstl
  LANGUAGES CXX
  VERSION 0.0.1)

# generate version header
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/xstl/version.hpp @ONLY)

set(GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/xstl")

add_library(xstl INTERFACE)
target_include_directories(
  xstl
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${GENERATED_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:include>)

# properly namespaced target
install(DIRECTORY include/ DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/xstl/version.hpp
        DESTINATION include/xstl)
install(TARGETS xstl EXPORT xstlTargets)
install(
  EXPORT xstlTargets
  FILE xstlTargets.cmake
  NAMESPACE xstl::
  DESTINATION lib/cmake/xstl)

set(xstl_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(xstl_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(xstl_VERSION_PATCH ${PROJECT_VERSION_PATCH})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/xstlConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/xstlConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/xstlConfig.cmake"
  INSTALL_DESTINATION lib/cmake/CLUEstering)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/xstlConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/xstlConfigVersion.cmake"
        DESTINATION lib/cmake/CLUEstering)
