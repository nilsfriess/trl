add_executable(csrtest_openmp csrtest.cc)
target_include_directories(csrtest_openmp PRIVATE ../../include ${CMAKE_BINARY_DIR}/_deps/spectra-src/include)
target_link_libraries(csrtest_openmp PRIVATE trl::trl Spectra)

