#pragma once

#include <cstdio>
#include <cstdlib>

#define NOT_IMPLEMENTED()                                                                                                                                                                              \
  do {                                                                                                                                                                                                 \
    std::fprintf(stderr, "%s:%d: error: Not implemented\n", __FILE__, __LINE__);                                                                                                                       \
    std::fflush(stderr);                                                                                                                                                                               \
    std::abort();                                                                                                                                                                                      \
  } while (0)

#define UNUSED(x) (void)(x)
