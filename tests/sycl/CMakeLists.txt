include(CTest)

set(TESTS
  test_lanczos_extend
  test_lanczos_convergence
)

if (${WITH_SYCL_BACKEND})
  foreach(test IN ITEMS ${TESTS})
    message(STATUS "Add test with name ${test}_sycl, file ${test}.cc")
    add_executable(${test}_sycl ${test}.cc)
    target_link_libraries(${test}_sycl PRIVATE trl::trl)
    target_include_directories(${test}_sycl PRIVATE ${PROJECT_SOURCE_DIR}/examples)
    target_compile_options(${test}_sycl PUBLIC -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter)
    add_sycl_to_target(TARGET ${test}_sycl)
    add_test(${test}_sycl ${test}_sycl)
  endforeach()
endif()
