add_executable(csrtest_openmp csrtest.cc)
target_include_directories(csrtest_openmp PRIVATE ../../include ${CMAKE_BINARY_DIR}/_deps/spectra-src/include)
target_link_libraries(csrtest_openmp PRIVATE trl::trl Spectra)

add_executable(gevp_test_openmp gevp_test.cc)
target_include_directories(gevp_test_openmp PRIVATE ../../include ${CMAKE_BINARY_DIR}/_deps/spectra-src/include)
target_link_libraries(gevp_test_openmp PRIVATE trl::trl Spectra)

