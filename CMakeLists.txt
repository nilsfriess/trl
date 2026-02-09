cmake_minimum_required(VERSION 3.16)
project(trl LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(WITH_SYCL_BACKEND "Enable the SYCL backend based" Off)

if(${WITH_SYCL_BACKEND})
  find_package(AdaptiveCpp REQUIRED)
endif()

# The OpenMP backend is always built
find_package(OpenMP REQUIRED)

# Create header-only library target
add_library(trl INTERFACE)
add_library(trl::trl ALIAS trl)

find_package(Eigen3 NO_MODULE REQUIRED)
target_link_libraries(trl INTERFACE Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
  target_link_libraries(trl INTERFACE OpenMP::OpenMP_CXX)
endif()

target_include_directories(trl INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if (${WITH_SYCL_BACKEND})
  target_compile_definitions(trl INTERFACE WITH_SYCL_BACKEND)
endif()


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
