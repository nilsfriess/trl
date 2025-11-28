#pragma once

#include <trl/impl/onemath/multivector.hh>

#include <oneapi/math.hpp>
#include <sycl/sycl.hpp>

template <class T, unsigned int bs>
class DiagonalEVP {
public:
  using Index = std::int64_t;
  using Scalar = T;
  static constexpr unsigned int blocksize = bs;
  using BlockMultivector = trl::BlockMultivector<T, bs>;
  using BlockView = typename BlockMultivector::BlockView;

  DiagonalEVP(sycl::queue queue, Index N)
      : queue(queue)
      , N(N)
  {
    // Diagonal matrix with entries: diag[i] = i+1 (simpler than (i+1)^2 to avoid large values)
    diag = sycl::malloc_shared<T>(N, queue);
    for (Index i = 0; i < N; ++i) diag[i] = static_cast<T>(i + 1);
  }

  ~DiagonalEVP()
  {
    sycl::free(diag, queue);
    if (ortho_scratchpad_initialized) sycl::free(ortho_scratchpad, queue);
  }

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

    // 2. Do a CPU-side Cholesky on R (lower-triangular)
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

    // 3. Compute V = V * L^{-T}
    oneapi::math::blas::row_major::trsm(queue, oneapi::math::side::right, oneapi::math::uplo::upper, oneapi::math::transpose::nontrans, oneapi::math::diag::nonunit, V.rows(), bs, Scalar(1.0), R.data,
                                        bs, V.data, bs)
        .wait();
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

  void solve_small_dense(const typename BlockMultivector::BlockMatrix& B, T* eigvals, typename BlockMultivector::BlockMatrix& eigvecs)
  {
    // B is a block tridiagonal matrix of dimension (block_rows * bs) x (block_cols * bs)
    // We need to compute eigenvalues and eigenvectors of this dense matrix

    const Index n = B.block_rows() * bs;

    // Allocate contiguous memory for the dense matrix (column-major for LAPACK)
    T* B_dense = sycl::malloc_shared<T>(n * n, queue);

    // Copy blocked matrix to contiguous column-major format using SYCL
    const T* B_data = B.data();
    const Index B_block_cols = B.block_cols();
    queue
        .parallel_for(sycl::range<2>(n, n),
                      [=](sycl::id<2> idx) {
                        Index row = idx[0];
                        Index col = idx[1];

                        // Determine which block this element belongs to
                        Index block_row = row / bs;
                        Index block_col = col / bs;
                        Index local_row = row % bs;
                        Index local_col = col % bs;

                        // Get value from blocked storage (row-major within blocks)
                        Index block_offset = (block_row * B_block_cols + block_col) * bs * bs;
                        T value = B_data[block_offset + local_row * bs + local_col];

                        // Store in column-major dense format
                        B_dense[col * n + row] = value;
                      })
        .wait();

    // Compute eigenvalues and eigenvectors using LAPACK syevd
    std::int64_t lda = n;
    std::int64_t scratchpad_size = oneapi::math::lapack::syevd_scratchpad_size<T>(queue, oneapi::math::job::vec, oneapi::math::uplo::lower, n, lda);

    T* scratchpad = sycl::malloc_shared<T>(static_cast<std::size_t>(scratchpad_size), queue);

    // Compute eigendecomposition (B_dense will be overwritten with eigenvectors)
    oneapi::math::lapack::syevd(queue, oneapi::math::job::vec, oneapi::math::uplo::lower, n, B_dense, lda, eigvals, scratchpad, scratchpad_size).wait();

    // Copy eigenvectors from column-major dense format back to blocked format
    T* eigvecs_data = eigvecs.data();
    const Index eigvecs_block_cols = eigvecs.block_cols();
    queue
        .parallel_for(sycl::range<2>(n, n),
                      [=](sycl::id<2> idx) {
                        Index row = idx[0];
                        Index col = idx[1];

                        // Determine which block this element belongs to
                        Index block_row = row / bs;
                        Index block_col = col / bs;
                        Index local_row = row % bs;
                        Index local_col = col % bs;

                        // Get value from column-major dense format
                        T value = B_dense[col * n + row];

                        // Store in blocked storage (row-major within blocks)
                        Index block_offset = (block_row * eigvecs_block_cols + block_col) * bs * bs;
                        eigvecs_data[block_offset + local_row * bs + local_col] = value;
                      })
        .wait();

    sycl::free(scratchpad, queue);
    sycl::free(B_dense, queue);
  }

private:
  sycl::queue queue;

  Index N;

  // Diagonal entries
  T* diag = nullptr;

  // Scratch space for orthonormalization
  T* ortho_scratchpad = nullptr;
  bool ortho_scratchpad_initialized = false;
};
