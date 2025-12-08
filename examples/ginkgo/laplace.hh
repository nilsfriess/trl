#pragma once

#include "evp_base.hh"

/// 1D Laplacian eigenvalue problem with Dirichlet boundary conditions
/// Matrix: tridiagonal with 2/h² on diagonal, -1/h² on off-diagonals
/// Exact eigenvalues: λ_k = (2/h²) * (1 - cos(k*π*h/(N*h+h))) for k = 1, 2, ..., N
/// For h=1: λ_k = 2 - 2*cos(k*π/(N+1))
template <class T, unsigned int bs>
class Laplace1DEigenproblem : public StandardGinkgoEVP<T, bs> {
public:
  using Base = StandardGinkgoEVP<T, bs>;
  using typename Base::BlockView;
  using typename Base::Scalar;

  Laplace1DEigenproblem(std::shared_ptr<const gko::Executor> exec, std::size_t N, T h = 1.0)
      : Base(exec, N)
      , h_(h)
  {
    if (N == 0) throw std::invalid_argument("Matrix size N must be positive");
    if (h <= 0) throw std::invalid_argument("Mesh spacing h must be positive");

    // Create 1D Laplacian matrix (tridiagonal with 2, -1, -1)
    // We use CSR format
    using IndexType = int;

    // We need to fill the matrix on the host first, then copy to device
    auto host_exec = this->exec_->get_master();

    // Create arrays for the CSR data
    gko::array<IndexType> row_ptrs(host_exec, N + 1);
    gko::array<IndexType> col_idxs(host_exec, 3 * N - 2);
    gko::array<T> values(host_exec, 3 * N - 2);

    auto* row_ptrs_data = row_ptrs.get_data();
    auto* col_idxs_data = col_idxs.get_data();
    auto* values_data = values.get_data();

    // Fill the CSR structure
    row_ptrs_data[0] = 0;
    IndexType idx = 0;
    T diag_val = 2.0 / (h_ * h_);
    T offdiag_val = -1.0 / (h_ * h_);

    for (std::size_t i = 0; i < N; ++i) {
      if (i > 0) {
        col_idxs_data[idx] = i - 1;
        values_data[idx] = offdiag_val;
        idx++;
      }
      col_idxs_data[idx] = i;
      values_data[idx] = diag_val;
      idx++;
      if (i < N - 1) {
        col_idxs_data[idx] = i + 1;
        values_data[idx] = offdiag_val;
        idx++;
      }
      row_ptrs_data[i + 1] = idx;
    }

    // Create CSR matrix from the arrays
    laplace_ = gko::matrix::Csr<T, IndexType>::create(this->exec_, gko::dim<2>{N, N}, std::move(values), std::move(col_idxs), std::move(row_ptrs));
  }

  void apply(BlockView x, BlockView y)
  {
    // Apply the Laplacian matrix to x and store in y
    laplace_->apply(x.data(), y.data());
  }

  T get_mesh_spacing() const { return h_; }

private:
  std::shared_ptr<gko::matrix::Csr<T, int>> laplace_;
  T h_; // Mesh spacing
};
