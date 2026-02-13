#pragma once

#include "multivector.hh"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

namespace trl::openmp {
template <class ScalarT, unsigned int block_size>
class EVPBase {
public:
  using Scalar = ScalarT;
  constexpr static auto blocksize = block_size;

  using BlockMultivector = ::trl::openmp::BlockMultivector<Scalar, blocksize>;
  using BlockView = typename BlockMultivector::BlockView;
  using BlockMatrix = typename BlockMultivector::BlockMatrix;
  using BlockMatrixBlockView = BlockMatrix::BlockView;

  explicit EVPBase(std::size_t n)
      : N(n)

  {
    if (N > 0) Vtemp.emplace(create_multivector(N, block_size));
  }

  virtual ~EVPBase() = default;

  virtual void apply(BlockView, BlockView) = 0;

  virtual void dot(BlockView V, BlockView W, BlockMatrixBlockView R) { V.dot(W, R); }

  virtual void orthonormalize(BlockView V, BlockMatrixBlockView R)
  {
    if constexpr (blocksize == 1) {
      this->dot(V, V, R);
      ScalarT norm = std::sqrt(R.data_[0]);
      for (std::size_t i = 0; i < N; ++i) V.data_[i] /= norm;
      R.data_[0] = norm;
    }
    else {
      // 1. Compute Gram matrix G = V^T * V (stored in R)
      this->dot(V, V, R);

      // 2. Compute Cholesky factorisation of Gram matrix
      // G = U^T * U where U is upper triangular
      Eigen::Map<Eigen::Matrix<ScalarT, blocksize, blocksize, Eigen::RowMajor>> RR(R.data_);
      Eigen::LLT<Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic>> llt(RR);
      if (llt.info() != Eigen::Success) throw std::runtime_error("Cholesky factorization failed in orthonormalize");
      RR = llt.matrixL().transpose();
      auto stored_R = RR.eval(); // Force evaluation into a separate matrix

      // 3. Compute U^{-1} and store in R temporarily
      RR = stored_R.inverse().eval(); // R now contains U^{-1}

      // 4. Compute V_new = V_old * U^{-1} using temporary to avoid aliasing
      auto Vtemp0 = Vtemp->block_view(0);
      V.mult(R, Vtemp0);   // Vtemp = V * U^{-1}
      V.copy_from(Vtemp0); // V = Vtemp

      // 5. Restore U in R (the Cholesky factor, not its inverse)
      // The Lanczos algorithm needs R such that V_old = V_new * R
      RR = stored_R;
    }
  }

  std::size_t size() const { return N; }

  BlockMultivector create_multivector(std::size_t rows, std::size_t cols) const { return BlockMultivector(rows, cols); }

  BlockMatrix create_blockmatrix(std::size_t block_rows, std::size_t block_cols) const { return BlockMatrix(block_rows, block_cols); }

  virtual std::size_t solve_small_dense(const BlockMatrix& B, BlockMatrixBlockView beta, std::size_t nev)
  {
    const auto n_total = B.block_rows() * blocksize;

    // Convert block matrix to dense Eigen matrix
    Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic> B_dense(n_total, n_total);

    for (std::size_t i = 0; i < B.block_rows(); ++i) {
      for (std::size_t j = 0; j < B.block_cols(); ++j) {
        auto block = B.block_view(i, j);
        for (unsigned int bi = 0; bi < blocksize; ++bi)
          for (unsigned int bj = 0; bj < blocksize; ++bj) B_dense(i * blocksize + bi, j * blocksize + bj) = block.data_[bi * blocksize + bj];
      }
    }

    // Compute eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic>> solver(B_dense);
    if (solver.info() != Eigen::Success) throw std::runtime_error("Eigendecomposition failed");

    // Store eigenvalues in descending order (largest first)
    // Eigen returns them in ascending order, so reverse
    std::vector<unsigned int> indices(n_total);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](const auto& i, const auto& j) { return std::abs(solver.eigenvalues()[i]) > std::abs(solver.eigenvalues()[j]); });

    eigenvalues.resize(n_total);
    for (std::size_t i = 0; i < n_total; ++i) eigenvalues[i] = solver.eigenvalues()(indices[i]);

    // Store eigenvectors in BlockMatrix format (also reversed to match eigenvalues)
    if (!eigenvectors) eigenvectors = std::make_unique<BlockMatrix>(B.block_rows(), B.block_cols());

    for (std::size_t i = 0; i < B.block_rows(); ++i) {
      for (std::size_t j = 0; j < B.block_cols(); ++j) {
        auto block = eigenvectors->block_view(i, j);
        for (unsigned int bi = 0; bi < blocksize; ++bi) {
          for (unsigned int bj = 0; bj < blocksize; ++bj) {
            // Reverse column order to match descending eigenvalue order
            block.data_[bi * blocksize + bj] = solver.eigenvectors()(i * blocksize + bi, indices[j * blocksize + bj]);
          }
        }
      }
    }

    // Compute residual norms and count converged eigenvalues
    // Residual norm: ||beta * v_j||_2 where v_j are the last blocksize components of eigenvector j
    std::size_t n_converged = 0;
    const std::size_t last_block_row = eigenvectors->block_rows() - 1;
    const ScalarT eps = std::numeric_limits<ScalarT>::epsilon();

    // std::cout << "  Residual norms: \n";
    const std::size_t n_check = std::min<std::size_t>(nev, n_total);
    for (std::size_t col_idx = 0; col_idx < n_check; ++col_idx) {
      const std::size_t block_col = col_idx / blocksize;
      const std::size_t col_in_block = col_idx % blocksize;

      auto v_last = eigenvectors->block_view(last_block_row, block_col);

      // Compute beta * v_j (where v_j is a column vector of size blocksize)
      ScalarT norm_sq = 0.0;
      for (unsigned int i = 0; i < blocksize; ++i) {
        ScalarT sum = 0.0;
        for (unsigned int k = 0; k < blocksize; ++k) sum += beta.data_[i * blocksize + k] * v_last.data_[k * blocksize + col_in_block];
        norm_sq += sum * sum;
      }

      ScalarT residual_norm = std::sqrt(norm_sq);
      const ScalarT theta = eigenvalues[col_idx];
      const ScalarT denom = std::max(std::abs(theta), eps);
      const ScalarT rel_residual = residual_norm / denom;

      if (rel_residual < tolerance_) n_converged++;
    }

    return n_converged;
  }

  const std::vector<Scalar>& get_current_eigenvalues() const { return eigenvalues; }

  const BlockMatrix& get_current_eigenvectors() const
  {
    assert(eigenvectors);
    return *eigenvectors;
  }

  std::span<Scalar, blocksize> get_eigenvalues_block(std::size_t block)
  {
    std::span<Scalar, blocksize> ev_block(eigenvalues.data() + block * blocksize, blocksize);
    return ev_block;
  }

  void set_tolerance(Scalar tol) { tolerance_ = tol; }
  Scalar get_tolerance() const { return tolerance_; }

protected:
  std::size_t N;

  std::optional<BlockMultivector> Vtemp; // Temporary for orthonormalize

  std::vector<Scalar> eigenvalues;
  std::unique_ptr<BlockMatrix> eigenvectors;

  Scalar tolerance_ = 1e-8;
};
} // namespace trl::openmp
