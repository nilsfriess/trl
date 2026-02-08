#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <optional>
#include <span>
#include <sycl/sycl.hpp>
#include <trl/impl/sycl/multivector.hh>

template <class T, unsigned int bs>
class StandardEVPBase {
public:
  using Index = std::int64_t;
  using Scalar = T;
  static constexpr unsigned int blocksize = bs;
  using BlockMultivector = trl::sycl::BlockMultivector<T, bs>;
  using BlockView = typename BlockMultivector::BlockView;

  /// Eigenvalue ordering for the projected system
  enum class EigenvalueOrder { Ascending, Descending };

  StandardEVPBase(sycl::queue queue, Index N)
      : queue(queue)
      , N(N)
  {
    if (N > 0) Vtemp.emplace(create_multivector(N, bs));
  }

  virtual ~StandardEVPBase() = default;

  virtual void apply(BlockView X, BlockView Y) = 0;

  void dot(BlockView X, BlockView Y, typename BlockMultivector::BlockMatrix::BlockView Z) { X.dot(Y, Z); }

  void orthonormalize(BlockView V, typename BlockMultivector::BlockMatrix::BlockView R)
  {
    // 1. Compute Gram matrix G = V^T * V (stored in R)
    V.dot(V, R);
    queue.wait();

    // 2. Compute Cholesky factorization of G = U^T * U
    Eigen::Map<Eigen::Matrix<T, bs, bs, Eigen::RowMajor>> RR(R.data);
    Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> llt(RR);
    if (llt.info() != Eigen::Success) throw std::runtime_error("Cholesky factorization failed in orthonormalize");
    RR = llt.matrixL().transpose();
    auto stored_R = RR.eval();

    // 3. Compute U^{-1} and store in R temporarily
    RR = stored_R.inverse().eval();

    auto Vtemp0 = Vtemp->block_view(0);
    V.mult(R, Vtemp0); // V_temp = V * U^{-1}
    V.copy_from(Vtemp0);
    queue.wait(); // Ensure copy completes

    // 4. Restore U in R (the Cholesky factor, not its inverse)
    RR = stored_R;
  }

  auto create_multivector(Index rows, Index cols) { return BlockMultivector(queue, rows, cols); }

  auto create_blockmatrix(Index block_rows, Index block_cols) { return typename BlockMultivector::BlockMatrix(queue, block_rows, block_cols); }

  std::size_t size() const { return static_cast<std::size_t>(N); }

  T* malloc(std::size_t n) const { return sycl::malloc_shared<T>(n, queue); }
  void free(T* ptr) const { sycl::free(ptr, queue); }

  std::vector<T> to_host_data(const T* ptr, std::size_t n)
  {
    queue.wait();
    std::vector<T> data(n);
    queue.memcpy(data.data(), ptr, n * sizeof(T)).wait();
    return data;
  }

  std::size_t solve_small_dense(const typename BlockMultivector::BlockMatrix& B, typename BlockMultivector::BlockMatrix::BlockView beta, std::size_t nev)
  {
    queue.wait();

    const auto n = B.block_rows() * bs;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B_dense(n, n);

    for (std::size_t i = 0; i < B.block_rows(); ++i) {
      for (std::size_t j = 0; j < B.block_cols(); ++j) {
        auto block = B.block_view(i, j);
        for (unsigned int bi = 0; bi < bs; ++bi)
          for (unsigned int bj = 0; bj < bs; ++bj) B_dense(i * bs + bi, j * bs + bj) = block(bi, bj);
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
    if (eigenvalues == nullptr) eigenvalues = sycl::malloc_shared<T>(n, queue);

    if (eigenvalue_order_ == EigenvalueOrder::Descending) {
      // Reverse order: largest eigenvalues first
      for (std::size_t i = 0; i < n; ++i) eigenvalues[i] = solver.eigenvalues()(n - 1 - i);
    }
    else {
      // Ascending order: smallest eigenvalues first (default)
      for (std::size_t i = 0; i < n; ++i) eigenvalues[i] = solver.eigenvalues()(i);
    }

    // Store eigenvectors in BlockMatrix format
    if (!eigenvectors) {
      eigenvectors = std::make_unique<typename BlockMultivector::BlockMatrix>(create_blockmatrix(B.block_rows(), B.block_cols()));
      queue.wait();
    }
    for (std::size_t i = 0; i < B.block_rows(); ++i) {
      for (std::size_t j = 0; j < B.block_cols(); ++j) {
        auto block = eigenvectors->block_view(i, j);
        const std::size_t n_cols = B.block_cols() * bs;
        for (unsigned int bi = 0; bi < bs; ++bi) {
          for (unsigned int bj = 0; bj < bs; ++bj) {
            if (eigenvalue_order_ == EigenvalueOrder::Descending) {
              // Reverse the column order for descending
              block(bi, bj) = solver.eigenvectors()(i * bs + bi, n_cols - 1 - (j * bs + bj));
            }
            else {
              // Normal order for ascending
              block(bi, bj) = solver.eigenvectors()(i * bs + bi, j * bs + bj);
            }
          }
        }
      }
    }

    // Compute the residual norms and return the number of converged eigenvalues
    const auto compute_norm = [&](std::size_t col_idx) -> T {
      // Compute ||beta * v_j||_2 where v_j are the last bs components of eigenvector col_idx
      // The last block row contains the last bs components
      const std::size_t last_block_row = eigenvectors->block_rows() - 1;
      auto v_last = eigenvectors->block_view(last_block_row, col_idx / bs);

      // Extract the column within the block
      const std::size_t col_in_block = col_idx % bs;

      // Compute beta * v_j (where v_j is a column vector of size bs)
      T norm_sq = 0.0;
      for (unsigned int i = 0; i < bs; ++i) {
        T sum = 0.0;
        for (unsigned int k = 0; k < bs; ++k) sum += beta(i, k) * v_last(k, col_in_block);
        norm_sq += sum * sum;
      }
      return std::sqrt(norm_sq);
    };

    // Count converged eigenvalues (residual norm < tolerance)
    const std::size_t n_eigs = eigenvectors->block_cols() * bs;
    const std::size_t n_check = std::min<std::size_t>(nev, n_eigs);
    std::size_t n_converged = 0;
    std::cout << "  Residual norms: \n";
    for (std::size_t j = 0; j < n_check; ++j) {
      T residual_norm = compute_norm(j);
      if (j < 16) std::cout << "    Eigenvalue " << j << ": " << residual_norm << "\n";
      if (residual_norm < tolerance_) n_converged++;
    }

    queue.wait();

    return n_converged;
  }

  std::span<T, std::dynamic_extent> get_current_eigenvalues() const { return std::span<T, std::dynamic_extent>(eigenvalues, eigenvectors->block_rows() * bs); }

  const typename BlockMultivector::BlockMatrix& get_current_eigenvectors() const
  {
    assert(eigenvectors);
    return *eigenvectors;
  }

  std::span<T, blocksize> get_eigenvalues_block(std::size_t block)
  {
    std::span<T, blocksize> ev_block(eigenvalues + block * blocksize, blocksize);
    return ev_block;
  }

  // Compute the column-wise 2-norms of the columns in B and return them on the host
  std::vector<T> two_norm_on_host(typename BlockMultivector::BlockMatrix::BlockView B)
  {
    queue.wait();
    std::vector<T> norms_host(bs, 0);

    for (std::size_t i = 0; i < bs; ++i)
      for (std::size_t j = 0; j < bs; ++j) norms_host[j] += B(i, j) * B(i, j);

    for (auto& n : norms_host) n = std::sqrt(n);
    return norms_host;
  }

  void set_tolerance(T tol) { tolerance_ = tol; }
  T get_tolerance() const { return tolerance_; }

protected:
  EigenvalueOrder eigenvalue_order_ = EigenvalueOrder::Descending;
  T tolerance_ = 1e-8;

  sycl::queue queue;

  Index N;

  std::optional<BlockMultivector> Vtemp;

  // Eigenvectors and eigenvalues
  T* eigenvalues = nullptr;
  std::unique_ptr<typename BlockMultivector::BlockMatrix> eigenvectors;
};
