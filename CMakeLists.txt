cmake_minimum_required(VERSION 3.16)
project(trl LANGUAGES C CXX VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(IntelSYCL CONFIG REQUIRED)
find_package(oneMath REQUIRED)

# Create header-only library target
add_library(trl INTERFACE)
add_library(trl::trl ALIAS trl)

target_include_directories(trl INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Only link oneMath, not IntelSYCL (which uses add_sycl_to_target instead)
target_link_libraries(trl INTERFACE
  ONEMATH::onemath
)

target_compile_features(trl INTERFACE cxx_std_20)

# Installation
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS trl
  EXPORT trlTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Generate and install package config files
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/trlConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/trlConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trl
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/trlConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/trlConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/trlConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trl
)

install(EXPORT trlTargets
  FILE trlTargets.cmake
  NAMESPACE trl::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trl
)

# Add tests and examples
enable_testing()
add_subdirectory(tests)
add_subdirectory(examples)
