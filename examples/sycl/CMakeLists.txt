if(${WITH_SYCL_BACKEND})
  add_executable(diagonal_sycl diagonal.cc)
  target_include_directories(diagonal_sycl PRIVATE ../../include)
  target_link_libraries(diagonal_sycl PRIVATE trl::trl)
  add_sycl_to_target(TARGET diagonal_sycl)
  target_precompile_headers(diagonal_sycl PRIVATE
    <sycl/sycl.hpp>
    <Eigen/Core>
    <Eigen/Dense>
    <Eigen/Sparse>
  )

  add_executable(csrtest_sycl csrtest.cc)
  target_include_directories(csrtest_sycl PRIVATE ../../include ${CMAKE_BINARY_DIR}/_deps/spectra-src/include)
  target_link_libraries(csrtest_sycl PRIVATE trl::trl Spectra)
  add_sycl_to_target(TARGET csrtest_sycl)

  add_executable(laptest_sycl laptest.cc)
  target_include_directories(laptest_sycl PRIVATE ../../include ${CMAKE_BINARY_DIR}/_deps/spectra-src/include)
  target_link_libraries(laptest_sycl PRIVATE trl::trl Spectra)
  add_sycl_to_target(TARGET laptest_sycl)

  add_executable(benchmarks_sycl benchmarks.cc)
  target_include_directories(benchmarks_sycl PRIVATE ../../include ${CMAKE_BINARY_DIR}/_deps/spectra-src/include)
  target_link_libraries(benchmarks_sycl PRIVATE trl::trl Spectra)
  add_sycl_to_target(TARGET benchmarks_sycl)

  add_executable(blockview_benchmarks_sycl blockview_benchmarks.cc)
  target_include_directories(blockview_benchmarks_sycl PRIVATE ../../include)
  target_link_libraries(blockview_benchmarks_sycl PRIVATE trl::trl)
  add_sycl_to_target(TARGET blockview_benchmarks_sycl)
  
  # target_link_libraries(laplace_example_sycl PRIVATE ONEMATH::onemath)
  # add_sycl_to_target(TARGET laplace_example_sycl SOURCES sycl/laplace.cc)
  # target_compile_definitions(laplace_example_sycl PUBLIC WITH_SYCL_BACKEND)
endif()
