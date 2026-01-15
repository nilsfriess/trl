#pragma once

#include "trl/impl/gingko/multivector.hh"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <ginkgo/ginkgo.hpp>

/// Base class for standard eigenvalue problems using Ginkgo backend
/// Provides common functionality for orthonormalization, small dense solves,
/// and eigenvalue/eigenvector storage
template <class T, unsigned int bs>
class StandardGinkgoEVP {
public:
  using Scalar = T;
  using BlockMultivector = trl::ginkgo::BlockMultivector<T, bs>;
  using BlockView = typename BlockMultivector::BlockView;
  using BlockMatrix = typename BlockMultivector::BlockMatrix;
  using MatrixBlockView = typename BlockMatrix::BlockView;
  static constexpr unsigned int blocksize = bs;

  /// Eigenvalue ordering for the projected system
  enum class EigenvalueOrder { Ascending, Descending };

  StandardGinkgoEVP(std::shared_ptr<const gko::Executor> exec, std::size_t N, EigenvalueOrder order = EigenvalueOrder::Ascending)
      : exec_(std::move(exec))
      , N_(N)
      , eigenvalue_order_(order)
      , current_eigenvalues_(exec_, 0)
      , current_eigenvectors_(exec_, 0, 0)
  {
  }

  virtual ~StandardGinkgoEVP() = default;

  /// Set eigenvalue ordering (Ascending for smallest first, Descending for largest first)
  void set_eigenvalue_order(EigenvalueOrder order) { eigenvalue_order_ = order; }

  void dot(BlockView x, BlockView y, MatrixBlockView B) { x.dot(y, B); }

  void orthonormalize(BlockView x, MatrixBlockView B)
  {
    this->dot(x, x, B);
    constexpr auto bsi = static_cast<int>(bs); // Store as int for Eigen to avoid warnings

    if (std::dynamic_pointer_cast<const gko::OmpExecutor>(exec_)) {

      // Map external buffers
      static constexpr auto Layout = (bsi == 1 ? Eigen::ColMajor : Eigen::RowMajor);
      using EigenMultiVec = Eigen::Matrix<T, Eigen::Dynamic, bsi, Layout>;
      using EigenSmallMatrix = Eigen::Matrix<T, bsi, bsi, Eigen::RowMajor>;
      Eigen::Map<EigenMultiVec> xmap(x.data()->get_values(), static_cast<Eigen::Index>(N_), bsi);
      Eigen::Map<EigenSmallMatrix> Rmap(B.data()->get_values());

      // Step 2: Cholesky: G = Rᵀ * R   (upper-triangular R)
      Eigen::LLT<Eigen::MatrixXd> llt(Rmap);
      if (llt.info() != Eigen::Success) throw std::runtime_error("Error in Eigen Cholesky");

      Rmap = llt.matrixU();
      xmap = xmap * Rmap.template triangularView<Eigen::Upper>().solve(Eigen::Matrix<T, bsi, bsi>::Identity());
    }
    else {
      throw std::runtime_error("orthonormalize is only implemented for the OMP executor");
    }
  }

  std::size_t size() const { return N_; }

  BlockMultivector create_multivector(std::size_t rows, std::size_t cols) const { return BlockMultivector(exec_, rows, cols); }

  BlockMatrix create_blockmatrix(std::size_t block_rows, std::size_t block_cols) const { return BlockMatrix(exec_, block_rows, block_cols); }

  void solve_small_dense(const BlockMatrix& B)
  {
    if (std::dynamic_pointer_cast<const gko::OmpExecutor>(exec_)) {
      const auto n = B.block_rows() * bs;
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B_dense(n, n);

      for (std::size_t i = 0; i < B.block_rows(); ++i) {
        for (std::size_t j = 0; j < B.block_cols(); ++j) {
          auto block = B.block_view(i, j);
          for (unsigned int bi = 0; bi < bs; ++bi)
            for (unsigned int bj = 0; bj < bs; ++bj) B_dense(i * bs + bi, j * bs + bj) = block.data()->at(bi, bj);
        }
      }

      // Compute eigendecomposition
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> solver(B_dense);
      if (solver.info() != Eigen::Success) {
        std::cerr << "Eigendecomposition info: " << (int)solver.info() << std::endl;
        throw std::runtime_error("Eigendecomposition failed");
      }

      // Get the eigenvalues from Eigen (in ascending order by default)
      // For Descending order (e.g., shift-invert), reverse to get largest first
      if (current_eigenvalues_.get_size() != n) current_eigenvalues_ = gko::array<T>(exec_, n);
      auto* eigvals_ptr = current_eigenvalues_.get_data();

      if (eigenvalue_order_ == EigenvalueOrder::Descending) {
        // Reverse order: largest eigenvalues first
        for (std::size_t i = 0; i < n; ++i) eigvals_ptr[i] = solver.eigenvalues()(n - 1 - i);
      }
      else {
        // Ascending order: smallest eigenvalues first (default)
        for (std::size_t i = 0; i < n; ++i) eigvals_ptr[i] = solver.eigenvalues()(i);
      }

      // Store eigenvectors in BlockMatrix format
      if (current_eigenvectors_.block_rows() != B.block_rows()) current_eigenvectors_ = create_blockmatrix(B.block_rows(), B.block_cols());
      for (std::size_t i = 0; i < B.block_rows(); ++i) {
        for (std::size_t j = 0; j < B.block_cols(); ++j) {
          auto block = current_eigenvectors_.block_view(i, j);
          const std::size_t n_cols = B.block_cols() * bs;
          for (unsigned int bi = 0; bi < bs; ++bi) {
            for (unsigned int bj = 0; bj < bs; ++bj) {
              if (eigenvalue_order_ == EigenvalueOrder::Descending) {
                // Reverse the column order for descending
                block.data()->at(bi, bj) = solver.eigenvectors()(i * bs + bi, n_cols - 1 - (j * bs + bj));
              }
              else {
                // Normal order for ascending
                block.data()->at(bi, bj) = solver.eigenvectors()(i * bs + bi, j * bs + bj);
              }
            }
          }
        }
      }
    }
    else {
      throw std::runtime_error("solve_small_dense is only implemented for the OMP executor");
    }
  }

  const gko::array<T>& get_current_eigenvalues() const { return current_eigenvalues_; }

  const gko::array<T>& get_eigenvalues_block(std::size_t block)
  {
    eigenvalues_block_ = gko::array<T>::view(exec_, bs, current_eigenvalues_.get_data() + block * bs);
    return eigenvalues_block_;
  }

  const gko::array<T>& get_temp_vector(std::size_t min_size)
  {
    if (temp_vector_.get_size() < min_size) temp_vector_ = gko::array<T>(exec_, min_size);
    return temp_vector_;
  }

  // Compute the column-wise 2-norms of the columns in B and return them on the host
  const std::vector<T>& two_norm_on_host(MatrixBlockView B)
  {
    norms_host_.resize(bs);
    std::fill(norms_host_.begin(), norms_host_.end(), 0);

    for (std::size_t i = 0; i < bs; ++i)
      for (std::size_t j = 0; j < bs; ++j) norms_host_[j] += B.data()->at(i, j) * B.data()->at(i, j);

    for (auto& n : norms_host_) n = std::sqrt(n);
    return norms_host_;
  }

  const BlockMatrix& get_current_eigenvectors() const { return current_eigenvectors_; }

protected:
  std::shared_ptr<const gko::Executor> exec_;
  std::size_t N_;
  EigenvalueOrder eigenvalue_order_;

  // Storage for eigenvalues and eigenvectors from solve_small_dense
  mutable gko::array<T> current_eigenvalues_;
  mutable BlockMatrix current_eigenvectors_;

  // Temporary storage that the eigensolver might request
  gko::array<T> temp_vector_;
  gko::array<T> eigenvalues_block_;
  mutable std::vector<T> norms_host_;
};
