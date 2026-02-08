include(FetchContent)
message(STATUS "Trying to download Spectra (C++ ARPACK alternative)")
FetchContent_Declare(
  Spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra
  GIT_TAG        v1.2.0
)
FetchContent_MakeAvailable(Spectra)

add_subdirectory(sycl)
add_subdirectory(openmp)

