include(CTest)

set(TESTS
  test_lanczos_extend
  test_lanczos_convergence
)

if (${WITH_GINKGO_BACKEND})
  find_package(Eigen3 REQUIRED)

  foreach(test IN ITEMS ${TESTS})
    message(STATUS "Add test with name ${test}_ginkgo, file ${test}.cc")
    add_executable(${test}_ginkgo ${test}.cc)
    target_link_libraries(${test}_ginkgo PRIVATE trl::trl Eigen3::Eigen)
    target_include_directories(${test}_ginkgo PRIVATE ${PROJECT_SOURCE_DIR}/examples)
    target_compile_options(${test}_ginkgo PUBLIC -Wall -Wextra -Wpedantic -Wshadow)
    add_test(${test}_ginkgo ${test}_ginkgo)
  endforeach()
endif()
