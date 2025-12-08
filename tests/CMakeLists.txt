# set(TESTS
#     test_diagonal_orthonormalize
#     test_lanczos_extend
#     test_eigenvalue_convergence
# )

# foreach(t ${TESTS})
#     add_executable(${t} ${t}.cc)
#     target_include_directories(${t} PRIVATE ../include ../examples)

#     if(${WITH_SYCL_BACKEND})
#       find_package(IntelSYCL REQUIRED)

#       target_link_libraries(${t} ONEMATH::onemath)
#       add_sycl_to_target(TARGET ${t} SOURCES ${t}.cc)
#       target_compile_definitions(${t} PUBLIC WITH_SYCL_BACKEND)
#     endif()

#     if(${WITH_GINKGO_BACKEND})
#       find_package (Eigen3 REQUIRED NO_MODULE)

#       target_compile_definitions(${t} PUBLIC WITH_GINKGO_BACKEND)
#       target_link_libraries(${t} Ginkgo::ginkgo Eigen3::Eigen)
#     endif()
    
#     add_test(NAME ${t} COMMAND ${t})
# endforeach()

add_subdirectory(ginkgo)
