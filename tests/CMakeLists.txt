set(TESTS
    test_diagonal_orthonormalize
    test_lanczos_extend
    test_eigenvalue_convergence
)

foreach(t ${TESTS})
    add_executable(${t} ${t}.cc)
    target_include_directories(${t} PRIVATE ../include ../examples)

    if(${WITH_SYCL_BACKEND})
      target_link_libraries(${t} PRIVATE ONEMATH::onemath)
      add_sycl_to_target(TARGET ${t} SOURCES ${t}.cc)
      target_compile_definitions(${t} PUBLIC WITH_SYCL_BACKEND)
    endif()

    if(${WITH_CUDA_BACKEND})
      target_compile_definitions(${t} PUBLIC WITH_CUDA_BACKEND)
    endif()
    
    add_test(NAME ${t} COMMAND ${t})
endforeach()
