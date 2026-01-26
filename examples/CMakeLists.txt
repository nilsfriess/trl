add_subdirectory(sycl)

if(${WITH_GINKGO_BACKEND})
  find_package (Eigen3 REQUIRED NO_MODULE)
  
  add_executable(diagonal_example_ginkgo ginkgo/diagonal.cc)
  target_include_directories(diagonal_example_ginkgo PRIVATE ../include)
  target_link_libraries(diagonal_example_ginkgo PRIVATE Ginkgo::ginkgo Eigen3::Eigen)
  target_compile_definitions(diagonal_example_ginkgo PUBLIC WITH_GINKGO_BACKEND)

  add_executable(laplace_example_ginkgo ginkgo/laplace.cc)
  target_include_directories(laplace_example_ginkgo PRIVATE ../include)
  target_link_libraries(laplace_example_ginkgo PRIVATE Ginkgo::ginkgo Eigen3::Eigen)
  target_compile_definitions(laplace_example_ginkgo PUBLIC WITH_GINKGO_BACKEND)

  add_executable(evp_examples_ginkgo ginkgo/evp_examples.cc)
  target_include_directories(evp_examples_ginkgo PRIVATE ../include)
  target_link_libraries(evp_examples_ginkgo PRIVATE Ginkgo::ginkgo Eigen3::Eigen)
  target_compile_definitions(evp_examples_ginkgo PUBLIC WITH_GINKGO_BACKEND)

  add_executable(generalized_fem_ginkgo ginkgo/generalized_fem.cc)
  target_include_directories(generalized_fem_ginkgo PRIVATE ../include)
  target_link_libraries(generalized_fem_ginkgo PRIVATE Ginkgo::ginkgo Eigen3::Eigen)
  target_compile_definitions(generalized_fem_ginkgo PUBLIC WITH_GINKGO_BACKEND)

endif()

