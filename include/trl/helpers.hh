#pragma once

#include <cstdlib>
#include <iostream>

inline void todo_impl(const char* file, int line, const char* message)
{
  std::cerr << file << ":" << line << ": TODO: " << message << std::endl;
  std::abort();
}

#define TRL_TODO(message) todo_impl(__FILE__, __LINE__, message)

inline void check_impl(const char* file, int line, bool condition, const char* message)
{
  if (!condition) [[unlikely]] {
    std::cerr << file << ":" << line << ": CHECK FAILED: " << message << std::endl;
    std::abort();
  }
}

#define TRL_CHECK(condition, message) check_impl(__FILE__, __LINE__, condition, message)
