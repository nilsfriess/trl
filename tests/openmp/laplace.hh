#pragma once

#include <cstddef>

#include "trl/impl/openmp/evp_base.hh"

namespace trl::openmp::tests {
template <class T, unsigned int bs>
class Laplace1DEVP : public trl::openmp::EVPBase<T, bs> {
public:
  using Base = trl::openmp::EVPBase<T, bs>;
  using BlockView = typename Base::BlockView;

  explicit Laplace1DEVP(std::size_t n)
      : Base(n)
  {
  }

  void apply(BlockView X, BlockView Y)
  {
    const std::size_t n = this->size();
#pragma omp parallel for
    for (std::size_t k = 0; k < n; ++k) {
      for (unsigned int i = 0; i < bs; ++i) {
        T val = T(2) * X.data[k * bs + i];
        if (k > 0) val -= X.data[(k - 1) * bs + i];
        if (k + 1 < n) val -= X.data[(k + 1) * bs + i];
        Y.data[k * bs + i] = val;
      }
    }
  }
};
} // namespace trl::openmp::tests
