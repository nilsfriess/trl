add_executable(laplace_example laplace.cc)
target_include_directories(laplace_example PRIVATE ../include)
target_link_libraries(laplace_example PRIVATE ONEMATH::onemath)
add_sycl_to_target(TARGET laplace_example SOURCES laplace.cc)