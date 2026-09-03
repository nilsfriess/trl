#pragma once

#include <cstddef>
#include <vector>

#include "trl/common.hh"
#include "trl/openmp/backend.hh"

template <class T, unsigned int bs>
class DiagonalEVPOperator : public trl::EuclideanDot<trl::openmp::Backend<T, bs>> {
public:
  using BlockView = typename trl::openmp::Backend<T, bs>::Multivector::BlockView;

  explicit DiagonalEVPOperator(std::size_t n)
      : diag(n)
  {
    for (std::size_t i = 0; i < n; ++i) diag[i] = static_cast<T>((i + 1) * (i + 1));
  }

  void apply(BlockView X, BlockView Y)
  {
#pragma omp parallel for
    for (std::size_t k = 0; k < diag.size(); ++k)
#pragma omp simd
      for (unsigned int i = 0; i < bs; ++i) Y.data_[k * bs + i] = diag[k] * X.data_[k * bs + i];
  }

  std::size_t size() const { return diag.size(); }

private:
  std::vector<T> diag;
};
