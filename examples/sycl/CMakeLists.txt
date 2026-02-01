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

  add_executable(csrtest csrtest.cc)
  target_include_directories(csrtest PRIVATE ../../include)
  target_link_libraries(csrtest PRIVATE trl::trl)
  add_sycl_to_target(TARGET csrtest)

  add_executable(benchmark_sycl benchmark.cc)
  target_include_directories(benchmark_sycl PRIVATE ../../include)
  target_link_libraries(benchmark_sycl PRIVATE trl::trl)
  add_sycl_to_target(TARGET benchmark_sycl)
  target_precompile_headers(benchmark_sycl PRIVATE
    <sycl/sycl.hpp>
  )
  
  # target_link_libraries(laplace_example_sycl PRIVATE ONEMATH::onemath)
  # add_sycl_to_target(TARGET laplace_example_sycl SOURCES sycl/laplace.cc)
  # target_compile_definitions(laplace_example_sycl PUBLIC WITH_SYCL_BACKEND)
endif()
