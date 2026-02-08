#pragma once

#include <cstddef>
#include <vector>

#include "trl/impl/openmp/evp_base.hh"

namespace trl::openmp::tests {
template <class T, unsigned int bs>
class DiagonalEVP : public trl::openmp::EVPBase<T, bs> {
public:
  using Base = trl::openmp::EVPBase<T, bs>;
  using BlockView = typename Base::BlockView;

  explicit DiagonalEVP(std::size_t n)
      : Base(n)
      , diag(n)
  {
    for (std::size_t i = 0; i < n; ++i) diag[i] = static_cast<T>((i + 1) * (i + 1));
  }

  void apply(BlockView X, BlockView Y)
  {
#pragma omp parallel for
    for (std::size_t k = 0; k < diag.size(); ++k)
#pragma omp simd
      for (unsigned int i = 0; i < bs; ++i) Y.data[k * bs + i] = diag[k] * X.data[k * bs + i];
  }

private:
  std::vector<T> diag;
};
} // namespace trl::openmp::tests
