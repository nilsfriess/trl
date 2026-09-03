include(FetchContent)
message(STATUS "Trying to download Spectra (C++ ARPACK alternative)")
FetchContent_Declare(
  Spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG        v1.2.0
)
FetchContent_MakeAvailable(Spectra)

include(${PROJECT_SOURCE_DIR}/cmake/trlBackendTargets.cmake)

add_subdirectory(sycl)
add_subdirectory(openmp)

