#pragma once

#include <cstddef>
#include <memory>

#include <sycl/sycl.hpp>

#include "trl/sycl/backend.hh"

#include "diagonal.hh"
#include "laplace.hh"

/** @brief Everything the shared test suites need that is backend-specific. */
struct Fixture {
  static constexpr const char* name = "SYCL";

  /// One in-order queue shared by the backend and every operator built here.
  static sycl::queue& queue()
  {
    static sycl::queue q{sycl::property::queue::in_order{}};
    return q;
  }

  template <class Scalar, unsigned int bs>
  using Backend = trl::Sycl::Backend<Scalar, bs>;

  template <class Scalar, unsigned int bs>
  static Backend<Scalar, bs> make_backend()
  {
    return Backend<Scalar, bs>(queue());
  }

  template <class Scalar, unsigned int bs>
  static auto make_diagonal(std::size_t n)
  {
    return std::make_shared<DiagonalEVPOperator<Scalar, bs>>(queue(), n);
  }

  template <class Scalar, unsigned int bs>
  static auto make_laplace(std::size_t n)
  {
    return std::make_shared<Laplace1DEVPOperator<Scalar, bs>>(queue(), n);
  }
};
