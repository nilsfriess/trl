#pragma once

#include "evp_base.hh"

/// Diagonal eigenvalue problem: A = diag(1, 2, 3, ..., N)
/// Exact eigenvalues: λ_i = i for i = 1, 2, ..., N
template <class T, unsigned int bs>
class DiagonalEigenproblem : public StandardGinkgoEVP<T, bs> {
public:
  using Base = StandardGinkgoEVP<T, bs>;
  using typename Base::BlockView;
  using typename Base::Scalar;

  DiagonalEigenproblem(std::shared_ptr<const gko::Executor> exec, std::size_t N)
      : Base(exec, N)
  {
    if (N == 0) throw std::invalid_argument("Matrix size N must be positive");

    diagonal_ = gko::matrix::Diagonal<T>::create(this->exec_, N);
    auto* values = diagonal_->get_values();
    for (std::size_t i = 0; i < N; ++i) values[i] = T(i + 1);
  }

  void apply(BlockView x, BlockView y)
  {
    // Apply the diagonal matrix to x and store in y: y = diag(1,2,3,...,N) * x
    diagonal_->apply(x.data(), y.data());
  }

private:
  std::shared_ptr<gko::matrix::Diagonal<T>> diagonal_;
};
