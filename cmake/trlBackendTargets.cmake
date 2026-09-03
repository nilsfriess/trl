# Helpers for the per-backend test and example targets.
#
# Every target differs only in the "_openmp"/"_sycl" suffix, whether SYCL has to
# be attached, and (for examples) whether Spectra is linked. These wrap that so
# the subdirectories stay a list of names.

# Common shape of any backend-specific target built from a single source file.
function(_trl_add_backend_target target backend source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE trl::trl)
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter)

  if(backend STREQUAL "sycl")
    add_sycl_to_target(TARGET ${target})
  endif()
endfunction()

# trl_add_backend_test(<name> BACKEND <openmp|sycl>)
#
# Builds <name>.cc as <name>_<backend> and registers it with CTest. SYCL targets
# are skipped unless WITH_SYCL_BACKEND is on.
function(trl_add_backend_test name)
  cmake_parse_arguments(ARG "" "BACKEND" "" ${ARGN})

  if(ARG_BACKEND STREQUAL "sycl" AND NOT WITH_SYCL_BACKEND)
    return()
  endif()

  set(target ${name}_${ARG_BACKEND})
  message(STATUS "Add test with name ${target}, file ${name}.cc")

  _trl_add_backend_target(${target} ${ARG_BACKEND} ${name}.cc)
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
  add_test(NAME ${target} COMMAND ${target})
endfunction()

# trl_add_backend_example(<name> BACKEND <openmp|sycl> [SPECTRA])
#
# Builds <name>.cc as <name>_<backend>. SPECTRA links the Spectra reference
# solver used by the comparison examples.
function(trl_add_backend_example name)
  cmake_parse_arguments(ARG "SPECTRA" "BACKEND" "" ${ARGN})

  if(ARG_BACKEND STREQUAL "sycl" AND NOT WITH_SYCL_BACKEND)
    return()
  endif()

  set(target ${name}_${ARG_BACKEND})

  _trl_add_backend_target(${target} ${ARG_BACKEND} ${name}.cc)
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/examples)

  if(ARG_SPECTRA)
    # Spectra is header-only. Pull its headers in as SYSTEM so its own warnings
    # do not drown out ours -- the examples are compiled with the same warning
    # flags as the tests.
    target_link_libraries(${target} PRIVATE Spectra)
    get_target_property(_spectra_includes Spectra INTERFACE_INCLUDE_DIRECTORIES)
    if(_spectra_includes)
      target_include_directories(${target} SYSTEM PRIVATE ${_spectra_includes})
    endif()
  endif()
endfunction()
