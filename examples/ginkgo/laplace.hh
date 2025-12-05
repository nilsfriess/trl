#pragma once

#include "trl/impl/gingko/multivector.hh"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <ginkgo/ginkgo.hpp>

// clang-format off
#define TODO(msg) do { \
    /* Emacs compilation buffer format: file:line: message */ \
    std::cerr << __FILE__ << ":" << __LINE__ << ": TODO in " << __func__ << ": " << msg << std::endl; \
    std::exit(EXIT_FAILURE); \
} while(0)
// clang-format on

template <class T, unsigned int bs>
class DiagonalEigenproblem {
public:
  using Scalar = T;
  using BlockMultivector = trl::ginkgo::BlockMultivector<T, bs>;
  using BlockView = typename BlockMultivector::BlockView;
  using BlockMatrix = typename BlockMultivector::BlockMatrix;
  using MatrixBlockView = typename BlockMatrix::BlockView;
  static constexpr unsigned int blocksize = bs;

  DiagonalEigenproblem(std::shared_ptr<const gko::Executor> exec, std::size_t N)
      : exec_(std::move(exec))
      , N_(N)
  {
    diagonal_ = gko::matrix::Diagonal<T>::create(exec_, N);
    auto* values = diagonal_->get_values();
    for (std::size_t i = 0; i < N; ++i) values[i] = T(i + 1);
  }

  void apply(BlockView x, BlockView y)
  {
    // Apply the diagonal matrix to x and store in y: y = diag(1,2,3,...,N) * x
    diagonal_->apply(x.data, y.data);
  }

  void dot(BlockView x, BlockView y, MatrixBlockView B) { x.dot(y, B); }

  void orthonormalize(BlockView x, MatrixBlockView B)
  {
    this->dot(x, x, B);

    if (std::dynamic_pointer_cast<const gko::OmpExecutor>(exec_)) {

      // Map external buffers
      static constexpr auto Layout = (bs == 1 ? Eigen::ColMajor : Eigen::RowMajor);
      using EigenMultiVec = Eigen::Matrix<T, Eigen::Dynamic, bs, Layout>;
      using EigenSmallMatrix = Eigen::Matrix<T, bs, bs, Eigen::RowMajor>;
      Eigen::Map<EigenMultiVec> xmap(x.data->get_values(), N_, bs);
      Eigen::Map<EigenSmallMatrix> Rmap(B.data->get_values());

      // Step 2: Cholesky: G = Rᵀ * R   (upper-triangular R)
      Eigen::LLT<Eigen::MatrixXd> llt(Rmap);
      if (llt.info() != Eigen::Success) std::runtime_error("Error in Eigen Cholesky");

      Rmap = llt.matrixU();
      xmap = xmap * Rmap.template triangularView<Eigen::Upper>().solve(Eigen::Matrix<T, bs, bs>::Identity());
    }
    else {
      throw std::runtime_error("orthonormalize is only implemented for the OMP executor");
    }
  }

  std::size_t size() const { return N_; }

  BlockMultivector create_multivector(std::size_t rows, std::size_t cols) const { return BlockMultivector(exec_, rows, cols); }

  BlockMatrix create_blockmatrix(std::size_t block_rows, std::size_t block_cols) const { return BlockMatrix(exec_, block_rows, block_cols); }

  // TODO: Implement other variants for the other executors
  void solve_small_dense(const BlockMatrix& B)
  {
    if (std::dynamic_pointer_cast<const gko::OmpExecutor>(exec_)) {
      const auto n = B.block_rows() * bs;
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B_dense(n, n);

      for (std::size_t i = 0; i < B.block_rows(); ++i) {
        for (std::size_t j = 0; j < B.block_cols(); ++j) {
          auto block = B.block_view(i, j);
          for (unsigned int bi = 0; bi < bs; ++bi)
            for (unsigned int bj = 0; bj < bs; ++bj) B_dense(i * bs + bi, j * bs + bj) = block.data->at(bi, bj);
        }
      }

      // Check if matrix is symmetric
      T max_asymmetry = 0;
      for (int i = 0; i < (int)n; ++i) {
        for (int j = i + 1; j < (int)n; ++j) {
          T asymmetry = std::abs(B_dense(i, j) - B_dense(j, i));
          max_asymmetry = std::max(max_asymmetry, asymmetry);
        }
      }
      std::cout << "Max asymmetry in matrix: " << max_asymmetry << std::endl;

      // Check diagonal and bounds
      T diag_min = B_dense.diagonal().minCoeff();
      T diag_max = B_dense.diagonal().maxCoeff();
      T norm = B_dense.norm();
      std::cout << "Diagonal min: " << diag_min << ", Diagonal max: " << diag_max << std::endl;
      std::cout << "Frobenius norm: " << norm << std::endl;

      // Compute eigendecomposition
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> solver(B_dense);
      if (solver.info() != Eigen::Success) {
        std::cerr << "Eigendecomposition info: " << (int)solver.info() << std::endl;
        throw std::runtime_error("Eigendecomposition failed");
      }
      else {
        // std::cout << "Solve small dense: Computed eigenvalues: " << solver.eigenvalues() << "\n";
      }

      std::vector<unsigned int> indices(n);
      std::iota(indices.begin(), indices.end(), 0);

      const auto& ev = solver.eigenvalues();
      // std::sort(indices.begin(), indices.end(), [&](auto i, auto j) { return ev[i] > ev[j]; });

      if (current_eigenvalues_.get_size() != n) current_eigenvalues_ = gko::array<T>(exec_, n);

      for (std::size_t i = 0; i < n; ++i) {
        current_eigenvalues_.get_data()[i] = ev[indices[i]];
        std::cout << i << ": " << ev[indices[i]] << "\n";
      }

      // auto* eigvals_ptr = current_eigenvalues_.get_data();
      // Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(eigvals_ptr, n) = solver.eigenvalues();

      // Store eigenvectors in BlockMatrix format
      if (current_eigenvectors_.block_rows() != B.block_rows()) current_eigenvectors_ = create_blockmatrix(B.block_rows(), B.block_cols());
      for (std::size_t i = 0; i < B.block_rows(); ++i) {
        for (std::size_t j = 0; j < B.block_cols(); ++j) {
          auto block = current_eigenvectors_.block_view(i, j);
          for (unsigned int bi = 0; bi < bs; ++bi)
            for (unsigned int bj = 0; bj < bs; ++bj) block.data->at(bi, bj) = solver.eigenvectors()(i * bs + bi, indices[j * bs + bj]);
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
      for (std::size_t j = 0; j < bs; ++j) norms_host_[j] += B.data->at(i, j) * B.data->at(i, j);

    for (auto& n : norms_host_) n = std::sqrt(n);
    return norms_host_;
  }

  const BlockMatrix& get_current_eigenvectors() const { return current_eigenvectors_; }

private:
  std::shared_ptr<const gko::Executor> exec_;
  std::size_t N_;

  using IndexType = int;
  using csr_matrix = gko::matrix::Csr<T, IndexType>;
  std::shared_ptr<gko::matrix::Diagonal<T>> diagonal_;

  // Storage for eigenvalues and eigenvectors from solve_small_dense
  mutable gko::array<T> current_eigenvalues_{exec_, 0};
  mutable BlockMatrix current_eigenvectors_{exec_, 0, 0};

  // Temporary storage that the eigensolver might request
  gko::array<T> temp_vector_;
  gko::array<T> eigenvalues_block_;
  mutable std::vector<T> norms_host_;
};
