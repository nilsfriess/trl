#pragma once

#include <cstddef>

#include "trl/common.hh"
#include "trl/openmp/backend.hh"

// Matrix-free 1D Laplacian: tridiagonal with 2 on the diagonal and -1 off it
// (h = 1, so no 1/h^2 scaling), with Dirichlet conditions X[-1] = X[n] = 0.
template <class T, unsigned int bs>
class Laplace1DEVPOperator : public trl::EuclideanDot<trl::openmp::Backend<T, bs>> {
public:
  using BlockView = typename trl::openmp::Backend<T, bs>::Multivector::BlockView;

  explicit Laplace1DEVPOperator(std::size_t n)
      : n_(n)
  {
  }

  void apply(BlockView X, BlockView Y)
  {
    const T* X_data = X.data();
    T* Y_data = Y.data();
    const std::size_t n = n_;

#pragma omp parallel for
    for (std::size_t k = 0; k < n; ++k) {
      for (unsigned int i = 0; i < bs; ++i) {
        T val = T(2) * X_data[k * bs + i];
        if (k > 0) val -= X_data[(k - 1) * bs + i];
        if (k + 1 < n) val -= X_data[(k + 1) * bs + i];
        Y_data[k * bs + i] = val;
      }
    }
  }

  std::size_t size() const { return n_; }

private:
  std::size_t n_;
};
