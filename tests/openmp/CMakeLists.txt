include(CTest)

set(TESTS
  test_lanczos_extend
  test_lanczos_convergence
)

foreach(test IN ITEMS ${TESTS})
  message(STATUS "Add test with name ${test}_openmp, file ${test}.cc")
  add_executable(${test}_openmp ${test}.cc)
  target_link_libraries(${test}_openmp PRIVATE trl::trl)
  target_include_directories(${test}_openmp PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
  target_compile_options(${test}_openmp PUBLIC -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter)
  add_test(${test}_openmp ${test}_openmp)
endforeach()
