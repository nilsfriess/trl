#pragma once

#include <cstddef>

#include "trl/common.hh"
#include "trl/openmp/backend.hh"

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
    const std::size_t n = size();
#pragma omp parallel for
    for (std::size_t k = 0; k < n; ++k) {
      for (unsigned int i = 0; i < bs; ++i) {
        T val = T(2) * X.data_[k * bs + i];
        if (k > 0) val -= X.data_[(k - 1) * bs + i];
        if (k + 1 < n) val -= X.data_[(k + 1) * bs + i];
        Y.data_[k * bs + i] = val;
      }
    }
  }

  std::size_t size() const { return n_; }

private:
  std::size_t n_;
};
