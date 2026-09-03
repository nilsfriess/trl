include(CTest)
include(${PROJECT_SOURCE_DIR}/cmake/trlBackendTargets.cmake)

add_subdirectory(sycl)
add_subdirectory(openmp)
