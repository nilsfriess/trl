#pragma once

#include <cstddef>
#include <memory>

#include "trl/openmp/backend.hh"

#include "diagonal.hh"
#include "laplace.hh"

/** @brief Everything the shared test suites need that is backend-specific. */
struct Fixture {
  static constexpr const char* name = "OpenMP";

  template <class Scalar, unsigned int bs>
  using Backend = trl::openmp::Backend<Scalar, bs>;

  template <class Scalar, unsigned int bs>
  static Backend<Scalar, bs> make_backend()
  {
    return {};
  }

  template <class Scalar, unsigned int bs>
  static auto make_diagonal(std::size_t n)
  {
    return std::make_shared<DiagonalEVPOperator<Scalar, bs>>(n);
  }

  template <class Scalar, unsigned int bs>
  static auto make_laplace(std::size_t n)
  {
    return std::make_shared<Laplace1DEVPOperator<Scalar, bs>>(n);
  }
};
