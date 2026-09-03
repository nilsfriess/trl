#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "trl/common.hh"
#include "trl/sycl/backend.hh"

// Matrix-free 1D Laplacian: tridiagonal with 2 on the diagonal and -1 off it
// (h = 1, so no 1/h^2 scaling), with Dirichlet conditions X[-1] = X[n] = 0.
// SYCL twin of tests/openmp/laplace.hh; only the kernel launch differs.
template <class T, unsigned int bs>
class Laplace1DEVPOperator : public trl::EuclideanDot<trl::Sycl::Backend<T, bs>> {
public:
  using BlockView = typename trl::Sycl::Backend<T, bs>::Multivector::BlockView;

  Laplace1DEVPOperator(sycl::queue queue, std::size_t n)
      : queue(queue)
      , n_(n)
  {
  }

  void apply(BlockView X, BlockView Y)
  {
    // Hoisted into locals: a device lambda cannot capture `this`.
    const std::size_t n = n_;
    const T* X_data = X.data;
    T* Y_data = Y.data;

    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
      const std::size_t k = idx[0];
      for (unsigned int i = 0; i < bs; ++i) {
        T val = T(2) * X_data[k * bs + i];
        if (k > 0) val -= X_data[(k - 1) * bs + i];
        if (k + 1 < n) val -= X_data[(k + 1) * bs + i];
        Y_data[k * bs + i] = val;
      }
    });
  }

  std::size_t size() const { return n_; }

private:
  sycl::queue queue;
  std::size_t n_;
};
