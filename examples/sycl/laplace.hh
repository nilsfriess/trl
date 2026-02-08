#pragma once

#include <sycl/sycl.hpp>

#include "evp_base.hh"

// Matrix-free 1D Laplacian EVP that inherits from StandardEVPBase
// (similar to DiagonalEVP, for use in tests)
template <class T, unsigned int bs>
class Laplace1DEVP : public StandardEVPBase<T, bs> {
public:
  using Base = StandardEVPBase<T, bs>;

  Laplace1DEVP(sycl::queue queue, typename Base::Index N)
      : Base(queue, N)
  {
  }

  void apply(typename Base::BlockView X, typename Base::BlockView Y)
  {
    // Matrix-free 1D Laplacian: Y = A * X
    // where A is tridiagonal with 2 on diagonal, -1 on off-diagonals
    // (using h = 1, so no scaling by 1/h²)
    // Dirichlet boundary conditions: X[-1] = X[N] = 0

    T* X_data = X.data;
    T* Y_data = Y.data;
    const std::size_t N = this->N;

    this->queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
      auto i = idx[0];
      for (std::size_t j = 0; j < bs; ++j) {
        T val = T(2) * X_data[i * bs + j];
        if (i > 0) val -= X_data[(i - 1) * bs + j];
        if (i < N - 1) val -= X_data[(i + 1) * bs + j];
        Y_data[i * bs + j] = val;
      }
    });
  }
};
