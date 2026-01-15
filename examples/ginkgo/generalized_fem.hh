#pragma once

#include "evp_base.hh"

#include <cmath>

/// Generalized FEM eigenvalue problem: A x = λ B x
/// where A is the stiffness matrix and B is the consistent mass matrix.
///
/// A = tridiag(-1, 2, -1)      (stiffness)
/// B = (1/6) * tridiag(1, 4, 1) (consistent mass)
///
/// Both matrices share the same eigenvectors: v_k(j) = sin(j*k*π/(n+1))
///
/// The generalized eigenvalues are:
///   λ_k = (2 - 2*cos(k*π/(n+1))) / ((1/6)*(4 + 2*cos(k*π/(n+1))))
///       = 6 * (1 - cos(k*π/(n+1))) / (2 + cos(k*π/(n+1)))
///
/// This class implements the shift-invert transformation:
///   (A - σB)^{-1} B x = μ x,  where μ = 1/(λ - σ)
///
/// For σ = 0, this finds eigenvalues closest to 0 (smallest eigenvalues).
template <class T, unsigned int bs>
class GeneralizedFEMEigenproblem : public StandardGinkgoEVP<T, bs> {
public:
  using Base = StandardGinkgoEVP<T, bs>;
  using typename Base::BlockView;
  using typename Base::Scalar;
  using typename Base::BlockMultivector;
  using typename Base::MatrixBlockView;
  using typename Base::EigenvalueOrder;

  GeneralizedFEMEigenproblem(std::shared_ptr<const gko::Executor> exec, std::size_t N, T sigma = 0.0)
      : Base(exec, N, EigenvalueOrder::Descending)  // Use descending order for shift-invert (largest μ first)
      , sigma_(sigma)
      , temp_(exec, N, bs)
  {
    if (N == 0) throw std::invalid_argument("Matrix size N must be positive");

    using IndexType = int;
    auto host_exec = this->exec_->get_master();

    // =========================================================================
    // Create stiffness matrix A = tridiag(-1, 2, -1)
    // =========================================================================
    {
      gko::array<IndexType> row_ptrs(host_exec, N + 1);
      gko::array<IndexType> col_idxs(host_exec, 3 * N - 2);
      gko::array<T> values(host_exec, 3 * N - 2);

      auto* row_ptrs_data = row_ptrs.get_data();
      auto* col_idxs_data = col_idxs.get_data();
      auto* values_data = values.get_data();

      row_ptrs_data[0] = 0;
      IndexType idx = 0;

      for (std::size_t i = 0; i < N; ++i) {
        if (i > 0) {
          col_idxs_data[idx] = i - 1;
          values_data[idx] = -1.0;
          idx++;
        }
        col_idxs_data[idx] = i;
        values_data[idx] = 2.0;
        idx++;
        if (i < N - 1) {
          col_idxs_data[idx] = i + 1;
          values_data[idx] = -1.0;
          idx++;
        }
        row_ptrs_data[i + 1] = idx;
      }

      A_ = gko::matrix::Csr<T, IndexType>::create(this->exec_, gko::dim<2>{N, N}, std::move(values), std::move(col_idxs), std::move(row_ptrs));
    }

    // =========================================================================
    // Create mass matrix B = (1/6) * tridiag(1, 4, 1)
    // =========================================================================
    {
      gko::array<IndexType> row_ptrs(host_exec, N + 1);
      gko::array<IndexType> col_idxs(host_exec, 3 * N - 2);
      gko::array<T> values(host_exec, 3 * N - 2);

      auto* row_ptrs_data = row_ptrs.get_data();
      auto* col_idxs_data = col_idxs.get_data();
      auto* values_data = values.get_data();

      row_ptrs_data[0] = 0;
      IndexType idx = 0;

      for (std::size_t i = 0; i < N; ++i) {
        if (i > 0) {
          col_idxs_data[idx] = i - 1;
          values_data[idx] = 1.0 / 6.0;
          idx++;
        }
        col_idxs_data[idx] = i;
        values_data[idx] = 4.0 / 6.0;
        idx++;
        if (i < N - 1) {
          col_idxs_data[idx] = i + 1;
          values_data[idx] = 1.0 / 6.0;
          idx++;
        }
        row_ptrs_data[i + 1] = idx;
      }

      B_ = gko::matrix::Csr<T, IndexType>::create(this->exec_, gko::dim<2>{N, N}, std::move(values), std::move(col_idxs), std::move(row_ptrs));
    }

    // =========================================================================
    // Create shifted matrix (A - σB) and factorize it
    // =========================================================================
    {
      gko::array<IndexType> row_ptrs(host_exec, N + 1);
      gko::array<IndexType> col_idxs(host_exec, 3 * N - 2);
      gko::array<T> values(host_exec, 3 * N - 2);

      auto* row_ptrs_data = row_ptrs.get_data();
      auto* col_idxs_data = col_idxs.get_data();
      auto* values_data = values.get_data();

      row_ptrs_data[0] = 0;
      IndexType idx = 0;

      // (A - σB) = tridiag(-1 - σ/6, 2 - 4σ/6, -1 - σ/6)
      T diag_val = 2.0 - sigma_ * (4.0 / 6.0);
      T offdiag_val = -1.0 - sigma_ * (1.0 / 6.0);

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

      shifted_ = gko::matrix::Csr<T, IndexType>::create(this->exec_, gko::dim<2>{N, N}, std::move(values), std::move(col_idxs), std::move(row_ptrs));
    }

    // Create the direct solver factory for (A - σB)
    auto solver_factory = gko::experimental::solver::Direct<T, IndexType>::build()
                              .with_factorization(gko::experimental::factorization::Lu<T, IndexType>::build().with_symbolic_algorithm(gko::experimental::factorization::symbolic_type::symmetric))
                              .on(this->exec_);

    // Generate the solver (performs LU factorization)
    solver_ = solver_factory->generate(shifted_);
  }

  /// Apply the shift-invert operator: y = (A - σB)^{-1} B x
  void apply(BlockView x, BlockView y)
  {
    // Step 1: temp = B * x
    auto temp_view = temp_.block_view(0);
    B_->apply(x.data(), temp_view.data());

    // Step 2: solve (A - σB) y = temp, i.e., y = (A - σB)^{-1} temp
    solver_->apply(temp_view.data(), y.data());
  }

  /// Compute the B-inner product: result = x^T B y
  void dot(BlockView x, BlockView y, MatrixBlockView result)
  {
    // temp = B * y
    auto temp_view = temp_.block_view(0);
    B_->apply(y.data(), temp_view.data());

    // result = x^T * temp
    x.dot(temp_view, result);
  }

  /// Orthonormalize with respect to the B-inner product
  void orthonormalize(BlockView x, MatrixBlockView R)
  {
    // Compute Gram matrix G = X^T B X
    this->dot(x, x, R);

    constexpr auto bsi = static_cast<int>(bs);
    const auto N = this->N_;

    if (std::dynamic_pointer_cast<const gko::OmpExecutor>(this->exec_)) {
      // Map external buffers
      static constexpr auto Layout = (bsi == 1 ? Eigen::ColMajor : Eigen::RowMajor);
      using EigenMultiVec = Eigen::Matrix<T, Eigen::Dynamic, bsi, Layout>;
      using EigenSmallMatrix = Eigen::Matrix<T, bsi, bsi, Eigen::RowMajor>;
      Eigen::Map<EigenMultiVec> xmap(x.data()->get_values(), static_cast<Eigen::Index>(N), bsi);
      Eigen::Map<EigenSmallMatrix> Rmap(R.data()->get_values());

      // Cholesky: G = R^T * R (upper-triangular R)
      Eigen::LLT<Eigen::MatrixXd> llt(Rmap);
      if (llt.info() != Eigen::Success) throw std::runtime_error("Error in Eigen Cholesky for B-orthonormalization");

      Rmap = llt.matrixU();
      xmap = xmap * Rmap.template triangularView<Eigen::Upper>().solve(Eigen::Matrix<T, bsi, bsi>::Identity());
    }
    else {
      throw std::runtime_error("orthonormalize is only implemented for the OMP executor");
    }
  }

  T get_sigma() const { return sigma_; }

  /// Convert shift-invert eigenvalue μ back to original eigenvalue λ
  /// μ = 1/(λ - σ)  =>  λ = σ + 1/μ
  T convert_eigenvalue(T mu) const { return sigma_ + 1.0 / mu; }

  /// Compute the exact generalized eigenvalue for mode k (0-indexed)
  /// λ_k = 6 * (1 - cos((k+1)*π/(n+1))) / (2 + cos((k+1)*π/(n+1)))
  static T exact_eigenvalue(std::size_t k, std::size_t N)
  {
    T theta = (k + 1) * M_PI / (N + 1);
    T cos_theta = std::cos(theta);
    return 6.0 * (1.0 - cos_theta) / (2.0 + cos_theta);
  }

private:
  std::shared_ptr<gko::matrix::Csr<T, int>> A_;       // Stiffness matrix
  std::shared_ptr<gko::matrix::Csr<T, int>> B_;       // Mass matrix
  std::shared_ptr<gko::matrix::Csr<T, int>> shifted_; // A - σB
  std::unique_ptr<gko::LinOp> solver_;                // Direct solver for (A - σB)^{-1}
  BlockMultivector temp_;                             // Temporary storage
  T sigma_;                                           // Shift parameter
};
