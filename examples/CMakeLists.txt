if(${WITH_SYCL_BACKEND})
  add_executable(laplace_example_sycl sycl/laplace.cc)
  target_include_directories(laplace_example_sycl PRIVATE ../include)
  
  target_link_libraries(laplace_example_sycl PRIVATE ONEMATH::onemath)
  add_sycl_to_target(TARGET laplace_example_sycl SOURCES sycl/laplace.cc)
  target_compile_definitions(laplace_example_sycl PUBLIC WITH_SYCL_BACKEND)
endif()

if(${WITH_GINKGO_BACKEND})
  add_executable(laplace_example_ginkgo ginkgo/laplace.cc)
  target_include_directories(laplace_example_ginkgo PRIVATE ../include)
  
  target_link_libraries(laplace_example_ginkgo PRIVATE Ginkgo::ginkgo)
  target_compile_definitions(laplace_example_ginkgo PUBLIC WITH_GINKGO_BACKEND)
endif()

