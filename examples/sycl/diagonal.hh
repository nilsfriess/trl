#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <span>
#include <sycl/sycl.hpp>
#include <trl/impl/sycl/multivector.hh>

template <class T, unsigned int bs>
class DiagonalEVP {
public:
  using Index = std::int64_t;
  using Scalar = T;
  static constexpr unsigned int blocksize = bs;
  using BlockMultivector = trl::sycl::BlockMultivector<T, bs>;
  using BlockView = typename BlockMultivector::BlockView;

  DiagonalEVP(sycl::queue queue, Index N)
      : queue(queue)
      , N(N)
      , eigenvectors(queue, 1, 1)
  {
    // Diagonal matrix with entries: diag[i] = i+1 (simpler than (i+1)^2 to avoid large values)
    diag = sycl::malloc_shared<T>(N, queue);
    for (Index i = 0; i < N; ++i) diag[i] = static_cast<T>(i + 1);
  }

  ~DiagonalEVP() { sycl::free(diag, queue); }

  void apply(BlockView X, BlockView Y)
  {
    // Y = D * X where D is diagonal
    const T* diag_ptr = diag;
    T* X_data = X.data;
    T* Y_data = Y.data;

    queue
        .parallel_for(sycl::range<2>(N, bs),
                      [=](sycl::id<2> idx) {
                        Index i = idx[0];
                        Index j = idx[1];
                        Y_data[i * bs + j] = diag_ptr[i] * X_data[i * bs + j];
                      })
        .wait();
  }

  void dot(BlockView X, BlockView Y, typename BlockMultivector::BlockMatrix::BlockView Z) { X.dot(Y, Z); }

  void orthonormalize(BlockView V, typename BlockMultivector::BlockMatrix::BlockView R)
  {
    // 1. Compute Gram matrix G = V^T * V (stored in R)
    V.dot(V, R);
    queue.wait();

    // 2. Do a CPU-side Cholesky on R
    // We compute G = U^T * U where U is upper triangular (stored in upper part of A)
    T* A = R.data;

    for (Index j = 0; j < bs; ++j) {
      // Diagonal element R[j,j]
      T sum = A[j * bs + j];
      for (Index k = 0; k < j; ++k) sum -= A[k * bs + j] * A[k * bs + j];

      if (sum <= 0.0) throw std::runtime_error("Matrix not SPD");
      A[j * bs + j] = std::sqrt(sum);

      // Elements to the right: R[j, i] for i > j
      for (Index i = j + 1; i < bs; ++i) {
        T s = A[j * bs + i];
        for (Index k = 0; k < j; ++k) s -= A[k * bs + i] * A[k * bs + j];
        A[j * bs + i] = s / A[j * bs + j];
      }
    }

    // Zero the lower triangle
    for (Index i = 0; i < bs; ++i)
      for (Index j = 0; j < i; ++j) A[i * bs + j] = 0.0;

    // Save U for later (R should output the Cholesky factor)
    Eigen::Matrix<T, bs, bs> U_saved;
    for (Index i = 0; i < bs; ++i)
      for (Index j = 0; j < bs; ++j) U_saved(i, j) = A[i * bs + j];

    // 3. Compute V = V * U^{-1}
    // We need U^{-1} for the multiplication: V_new = V_old * U^{-1}
    // Due to Eigen's column-major and our row-major, the mapping transposes:
    // Eigen sees A^T. So if A contains U (upper tri), Eigen sees U^T (lower tri).
    Eigen::Map<Eigen::Matrix<T, bs, bs>> Lt(A);
    auto Lti = Lt.inverse().eval(); // Computes (U^T)^{-1} = U^{-T}
    Lt = Lti;                       // A now contains U^{-T} in Eigen's view
                                    // In our row-major view, A contains (U^{-T})^T = U^{-1}

    // Use temporary to avoid aliasing
    auto V_temp = create_multivector(V.rows(), bs);
    auto V_temp_block = V_temp.block_view(0);
    V.mult(R, V_temp_block); // V_temp = V * U^{-1}
    queue.wait();            // Wait for mult to complete
    V.copy_from(V_temp_block);
    queue.wait(); // Ensure copy completes

    // 4. Restore U in R (the Cholesky factor, not its inverse)
    // The Lanczos algorithm needs R such that V_old = V_new * R
    // Since V_old = V_new * U, we need R = U
    for (Index i = 0; i < bs; ++i)
      for (Index j = 0; j < bs; ++j) A[i * bs + j] = U_saved(i, j);
  }

  auto create_multivector(Index rows, Index cols) { return BlockMultivector(queue, rows, cols); }

  auto create_blockmatrix(Index block_rows, Index block_cols) { return typename BlockMultivector::BlockMatrix(queue, block_rows, block_cols); }

  std::size_t size() const { return static_cast<std::size_t>(N); }

  T* malloc(std::size_t n) const { return sycl::malloc_shared<T>(n, queue); }
  void free(T* ptr) const { sycl::free(ptr, queue); }

  std::vector<T> to_host_data(const T* ptr, std::size_t n)
  {
    std::vector<T> data(n);
    queue.memcpy(data.data(), ptr, n * sizeof(T)).wait();
    return data;
  }

  void solve_small_dense(const typename BlockMultivector::BlockMatrix& B) {
    assert(false && "not implemented");
  }

  std::span<T, std::dynamic_extent> get_current_eigenvalues() const
  {
    assert(false && "not implemented");
    return std::span<T, std::dynamic_extent>(eigenvalues, eigenvectors.block_rows() * bs);
  }

  const typename BlockMultivector::BlockMatrix& get_current_eigenvectors() const
  {
    assert(false && "not implemented");
    return eigenvectors;
  }

  std::span<T, blocksize> get_eigenvalues_block(std::size_t block)
  {
    std::span<T, blocksize> ev_block(eigenvalues + block * blocksize, blocksize);
    return ev_block;
  }

  // Compute the column-wise 2-norms of the columns in B and return them on the host
  std::vector<T> two_norm_on_host(typename BlockMultivector::BlockMatrix::BlockView B)
  {
    assert(false && "not implemented");
    return {};
  }

private:
  sycl::queue queue;

  Index N;

  // Diagonal entries
  T* diag = nullptr;

  // Eigenvectors and eigenvalues
  T* eigenvalues;
  typename BlockMultivector::BlockMatrix eigenvectors;
};
