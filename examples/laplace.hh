#pragma once

#include <trl/impl/onemath/multivector.hh>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <oneapi/math.hpp>
#include <stdexcept>
#include <sycl/sycl.hpp>

namespace trl {

template <class T, unsigned int bs>
class OneDLaplaceEVP {
public:
  using Index = std::int64_t;
  using Scalar = T;
  static constexpr unsigned int blocksize = bs;
  using BlockMultivector = trl::BlockMultivector<T, bs>;
  using BlockView = typename BlockMultivector::BlockView;

  OneDLaplaceEVP(sycl::queue queue, Index N)
      : queue(queue)
      , N(N)
  {
    Index nnz = (N - 2) * 3 + 2 + 2; // 3 entries in all rows except the first and the last

    // Initialise the CSR data for a 1d Laplacian
    row_ptr = sycl::malloc_shared<Index>(N + 1, queue);
    col_ind = sycl::malloc_shared<Index>(nnz, queue);
    val = sycl::malloc_shared<T>(nnz, queue);

    row_ptr[0] = 0;
    Index cnt = 0;
    for (Index i = 0; i < N; ++i) {
      if (i > 0) {
        col_ind[cnt] = i - 1;
        val[cnt++] = -1;
      }

      col_ind[cnt] = i;
      // Dirichlet BC: diagonal is 2 for all rows
      val[cnt++] = 2;

      if (i < N - 1) {
        col_ind[cnt] = i + 1;
        val[cnt++] = -1;
      }

      row_ptr[i + 1] = cnt;
    }
    assert(row_ptr[N] == nnz);

    oneapi::math::sparse::init_csr_matrix(queue, &A, N, N, nnz, oneapi::math::index_base::zero, row_ptr, col_ind, val);

    // Initialise the handles for the SpMM product
    oneapi::math::sparse::init_spmm_descr(queue, &spmm);
  }

  ~OneDLaplaceEVP()
  {
    oneapi::math::sparse::release_sparse_matrix(queue, A).wait();
    sycl::free(row_ptr, queue);
    sycl::free(col_ind, queue);
    sycl::free(val, queue);
    if (ortho_scratchpad_initialized) sycl::free(ortho_scratchpad, queue);
    if (spmm_initialised) {
      oneapi::math::sparse::release_dense_matrix(queue, X_dense).wait();
      oneapi::math::sparse::release_dense_matrix(queue, Y_dense).wait();
      sycl::free(workspace, queue);
    }
    oneapi::math::sparse::release_spmm_descr(queue, spmm).wait();
  }

  void apply(BlockView X, BlockView Y)
  {
    if (!spmm_initialised) {
      spmm_initialised = true;

      oneapi::math::sparse::init_dense_matrix<T>(queue, &X_dense, N, bs, bs, oneapi::math::layout::row_major, X.data);
      oneapi::math::sparse::init_dense_matrix<T>(queue, &Y_dense, N, bs, bs, oneapi::math::layout::row_major, Y.data);
      oneapi::math::sparse::spmm_buffer_size(queue, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, &alpha, A_view, A, X_dense, &beta, Y_dense, alg, spmm, temp_buffer_size);
      workspace = sycl::malloc_shared<T>(temp_buffer_size, queue);
      oneapi::math::sparse::spmm_optimize(queue, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, &alpha, A_view, A, X_dense, &beta, Y_dense, alg, spmm, workspace).wait();
    }

    oneapi::math::sparse::set_dense_matrix_data(queue, X_dense, N, bs, bs, oneapi::math::layout::row_major, X.data);
    oneapi::math::sparse::set_dense_matrix_data(queue, Y_dense, N, bs, bs, oneapi::math::layout::row_major, Y.data);

    oneapi::math::sparse::spmm(queue, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, &alpha, A_view, A, X_dense, &beta, Y_dense, alg, spmm).wait();
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

  std::size_t size() const { return N; }

  BlockMultivector create_multivector(std::size_t rows, std::size_t cols) { return BlockMultivector(queue, rows, cols); }

  typename BlockMultivector::BlockMatrix create_blockmatrix(std::size_t block_rows, std::size_t block_cols) { return typename BlockMultivector::BlockMatrix(queue, block_rows, block_cols); }

  T* malloc(std::size_t n) const { return sycl::malloc_shared<T>(n, queue); }
  void free(T* ptr) const { sycl::free(ptr, queue); }

  void solve_small_dense(const typename BlockMultivector::BlockMatrix& B, T* eigvals, typename BlockMultivector::BlockMatrix& eigvecs)
  {
    // B is a block tridiagonal matrix of dimension (block_rows * bs) x (block_cols * bs)
    // We need to compute eigenvalues and eigenvectors of this dense matrix

    // Allocate contiguous memory for the dense matrix (column-major for LAPACK)
    const Index n = B.block_rows() * bs;
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

  // Handles of the matrix and CSR data
  oneapi::math::sparse::matrix_handle_t A;
  Index* row_ptr = nullptr;
  Index* col_ind = nullptr;
  T* val = nullptr;

  // Handles for the matrix-vector product
  oneapi::math::sparse::spmm_descr_t spmm;
  T alpha = 1;
  T beta = 0;
  oneapi::math::sparse::matrix_view A_view{}; // Default values are ok
  oneapi::math::sparse::dense_matrix_handle_t X_dense;
  oneapi::math::sparse::dense_matrix_handle_t Y_dense;
  oneapi::math::sparse::spmm_alg alg = oneapi::math::sparse::spmm_alg::default_alg;
  std::size_t temp_buffer_size;
  T* workspace;
  bool spmm_initialised = false;

  // Scratchpad for orthonormalization
  T* ortho_scratchpad = nullptr;
  std::int64_t ortho_scratchpad_size = 0;
  bool ortho_scratchpad_initialized = false;
};

} // namespace trl
