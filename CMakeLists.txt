cmake_minimum_required(VERSION 3.16)
project(dd_benchmark LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(IntelSYCL CONFIG REQUIRED)

find_package(oneMath REQUIRED)

# Add tests and examples
enable_testing()
add_subdirectory(tests)
add_subdirectory(examples)
