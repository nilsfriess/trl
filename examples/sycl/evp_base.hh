#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
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

    if constexpr (bs == 1) {
      // R.data is device memory; copy the single scalar to host to compute the norm.
      T norm_sq;
      queue.memcpy(&norm_sq, R.data, sizeof(T)).wait();
      const T norm = std::sqrt(norm_sq);
      // Write norm back as the bs=1 "Cholesky factor" (async; in-order queue)
      queue.memcpy(R.data, &norm, sizeof(T));
      T* v_data = V.data;
      const std::size_t v_rows = V.rows();
      queue.parallel_for(sycl::range<1>{v_rows}, [=](sycl::id<1> idx) {
        v_data[idx[0]] /= norm;
      });
    }
    else {
      // 2. Host-stage R (bs×bs), compute Cholesky, triangular solve, copy back.
      T R_host[bs * bs];
      queue.memcpy(R_host, R.data, bs * bs * sizeof(T)).wait();

      Eigen::Map<Eigen::Matrix<T, bs, bs, Eigen::RowMajor>> RR(R_host);
      Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> llt(RR);
      if (llt.info() != Eigen::Success) throw std::runtime_error("Cholesky factorization failed in orthonormalize");
      RR = llt.matrixL().transpose();
      const Eigen::Matrix<T, bs, bs, Eigen::RowMajor> stored_R = RR;

      // 3. Compute U^{-1} via a triangular solve and store it in R temporarily.
      Eigen::Matrix<T, bs, bs, Eigen::RowMajor> inv_R = Eigen::Matrix<T, bs, bs, Eigen::RowMajor>::Identity();
      stored_R.template triangularView<Eigen::Upper>().solveInPlace(inv_R);
      RR = inv_R;

      // Copy inv_R to device so the gemm can use it.
      queue.memcpy(R.data, R_host, bs * bs * sizeof(T)).wait();

      auto Vtemp0 = Vtemp->block_view(0);
      Vtemp0.template gemm<false>(1., V, R, 0.); // V_temp = V * U^{-1}
      V.copy_from(Vtemp0);
      queue.wait();

      // 4. Restore U in R (the Cholesky factor, not its inverse)
      queue.memcpy(R.data, stored_R.data(), bs * bs * sizeof(T));
    }
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

  std::size_t solve_small_dense(const typename BlockMultivector::BlockMatrix& B, typename BlockMultivector::BlockMatrix::BlockView beta,
                                std::size_t nev)
  {
    queue.wait();

    const auto n = B.block_rows() * bs;

    // Copy B matrix from device to host
    const std::size_t B_total = B.block_rows() * B.block_cols() * bs * bs;
    std::vector<T> B_host(B_total);
    queue.memcpy(B_host.data(), B.data(), B_total * sizeof(T)).wait();

    // Copy beta from device to host
    T beta_host[bs * bs];
    queue.memcpy(beta_host, beta.data, bs * bs * sizeof(T)).wait();

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B_dense(n, n);

    for (std::size_t i = 0; i < B.block_rows(); ++i)
      for (std::size_t j = 0; j < B.block_cols(); ++j)
        for (unsigned int bi = 0; bi < bs; ++bi)
          for (unsigned int bj = 0; bj < bs; ++bj)
            B_dense(i * bs + bi, j * bs + bj) =
                B_host[(i * B.block_cols() + j) * bs * bs + bi * bs + bj];

    // Compute eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> solver(B_dense);
    if (solver.info() != Eigen::Success) {
      std::cerr << "Eigendecomposition info: " << (int)solver.info() << std::endl;
      throw std::runtime_error("Eigendecomposition failed");
    }

    // Get the eigenvalues from Eigen
    std::vector<unsigned int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](const auto& i, const auto& j) { return std::abs(solver.eigenvalues()[i]) > std::abs(solver.eigenvalues()[j]); });
    eigenvalues.resize(n);
    for (std::size_t i = 0; i < n; ++i) eigenvalues[i] = solver.eigenvalues()(indices[i]);

    const auto& ritz_vectors = solver.eigenvectors();

    // Compute the residual norms directly from Eigen's host-side Ritz vectors.
    const auto compute_norm = [&](std::size_t col_idx) -> T {
      const std::size_t last_block_row = B.block_rows() - 1;
      const std::size_t eigvec_col = indices[col_idx];

      T norm_sq = 0.0;
      for (unsigned int i = 0; i < bs; ++i) {
        T sum = 0.0;
        for (unsigned int k = 0; k < bs; ++k) sum += beta_host[i * bs + k] * ritz_vectors(last_block_row * bs + k, eigvec_col);
        norm_sq += sum * sum;
      }
      return std::sqrt(norm_sq);
    };

    // Count converged eigenvalues using the same relative residual criterion as OpenMP.
    const std::size_t n_eigs = B.block_cols() * bs;
    const std::size_t n_check = std::min<std::size_t>(nev, n_eigs);
    std::size_t n_converged = 0;
    const T eps = std::numeric_limits<T>::epsilon();
    for (std::size_t j = 0; j < n_check; ++j) {
      T residual_norm = compute_norm(j);
      const T theta = eigenvalues[j];
      const T denom = std::max(std::abs(theta), eps);
      const T rel_residual = residual_norm / denom;
      if (rel_residual < tolerance_) n_converged++;
    }

    // Only materialize Ritz vectors in backend storage if the caller will restart.
    if (n_converged < nev) {
      if (!eigenvectors || eigenvectors->block_rows() != B.block_rows() || eigenvectors->block_cols() != B.block_cols())
        eigenvectors = std::make_unique<typename BlockMultivector::BlockMatrix>(create_blockmatrix(B.block_rows(), B.block_cols()));

      // Build eigenvector data in a host buffer, then copy to device in one shot.
      std::vector<T> eig_host(B_total);
      for (std::size_t i = 0; i < B.block_rows(); ++i)
        for (std::size_t j = 0; j < B.block_cols(); ++j)
          for (unsigned int bi = 0; bi < bs; ++bi)
            for (unsigned int bj = 0; bj < bs; ++bj)
              eig_host[(i * B.block_cols() + j) * bs * bs + bi * bs + bj] =
                  ritz_vectors(i * bs + bi, indices[j * bs + bj]);

      queue.memcpy(eigenvectors->data(), eig_host.data(), B_total * sizeof(T)).wait();
    }

    return n_converged;
  }

  std::span<const T, std::dynamic_extent> get_current_eigenvalues() const
  {
    return std::span<const T, std::dynamic_extent>(eigenvalues.data(), eigenvalues.size());
  }

  const typename BlockMultivector::BlockMatrix& get_current_eigenvectors() const
  {
    assert(eigenvectors);
    return *eigenvectors;
  }

  std::span<T, blocksize> get_eigenvalues_block(std::size_t block)
  {
    std::span<T, blocksize> ev_block(eigenvalues.data() + block * blocksize, blocksize);
    return ev_block;
  }

  // Compute the column-wise 2-norms of the columns in B and return them on the host
  std::vector<T> two_norm_on_host(typename BlockMultivector::BlockMatrix::BlockView B)
  {
    queue.wait();
    T B_host[bs * bs];
    queue.memcpy(B_host, B.data, bs * bs * sizeof(T)).wait();
    std::vector<T> norms_host(bs, 0);
    for (std::size_t i = 0; i < bs; ++i)
      for (std::size_t j = 0; j < bs; ++j) norms_host[j] += B_host[i * bs + j] * B_host[i * bs + j];
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

  // Eigenvectors live in backend storage for restart operations; eigenvalues stay on the host.
  std::vector<T> eigenvalues;
  std::unique_ptr<typename BlockMultivector::BlockMatrix> eigenvectors;
};
