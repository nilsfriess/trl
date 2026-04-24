#pragma once

#include "evp_base.hh"

#include <algorithm>
#include <cusparse.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <trl/impl/sycl/profiling.hh>
#include <vector>

// CSR (Compressed Sparse Row) matrix storage
template <class Scalar>
struct CSRMatrix {
  int num_rows;
  int num_cols;
  int num_nonzeros;
  int* row_offsets; // size: num_rows + 1
  int* col_indices; // size: num_nonzeros
  Scalar* values;   // size: num_nonzeros
};

// Read a Matrix Market file and assemble into CSR format using USM
template <class Scalar>
inline CSRMatrix<Scalar> load_matrix_market(const std::string& filepath, sycl::queue& queue)
{
  std::ifstream file(filepath);
  if (!file) throw std::runtime_error("Error: cannot open file " + filepath);

  // Parse the Matrix Market header to check for symmetry
  bool is_symmetric = false;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    if (line[0] == '%') {
      // Check header line for symmetry indicator
      // Format: %%MatrixMarket matrix coordinate real symmetric
      if (line.find("%%MatrixMarket") != std::string::npos || line.find("%%matrixmarket") != std::string::npos) {
        // Convert to lowercase for case-insensitive comparison
        std::string lower_line = line;
        std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);
        if (lower_line.find("symmetric") != std::string::npos) is_symmetric = true;
      }
      continue;
    }
    // First non-comment line contains dimensions
    break;
  }

  // Read matrix dimensions and number of nonzeros from the current line
  int num_rows, num_cols, num_entries_in_file;
  std::istringstream iss(line);
  iss >> num_rows >> num_cols >> num_entries_in_file;

  // First pass: read COO data and count the actual number of nonzeros
  // (symmetric matrices need to add mirror entries for off-diagonal elements)
  std::vector<int> coo_rows;
  std::vector<int> coo_cols;
  std::vector<Scalar> coo_vals;

  coo_rows.reserve(is_symmetric ? 2 * num_entries_in_file : num_entries_in_file);
  coo_cols.reserve(is_symmetric ? 2 * num_entries_in_file : num_entries_in_file);
  coo_vals.reserve(is_symmetric ? 2 * num_entries_in_file : num_entries_in_file);

  for (int i = 0; i < num_entries_in_file; i++) {
    int row, col;
    Scalar val;
    file >> row >> col >> val;
    row--; // Convert 1-based to 0-based indexing
    col--;

    coo_rows.push_back(row);
    coo_cols.push_back(col);
    coo_vals.push_back(val);

    // For symmetric matrices, add the mirror entry (j, i) if i != j
    if (is_symmetric && row != col) {
      coo_rows.push_back(col);
      coo_cols.push_back(row);
      coo_vals.push_back(val);
    }
  }

  const int num_nonzeros = static_cast<int>(coo_rows.size());

  // Build CSR structure in host staging buffers
  int* h_row_offsets = sycl::malloc_host<int>(num_rows + 1, queue);
  int* h_col_indices = sycl::malloc_host<int>(num_nonzeros, queue);
  Scalar* h_values = sycl::malloc_host<Scalar>(num_nonzeros, queue);

  for (int i = 0; i <= num_rows; i++) h_row_offsets[i] = 0;
  for (int i = 0; i < num_nonzeros; i++) h_row_offsets[coo_rows[i] + 1]++;
  for (int i = 1; i <= num_rows; i++) h_row_offsets[i] += h_row_offsets[i - 1];

  std::vector<int> insert_pos(h_row_offsets, h_row_offsets + num_rows);
  for (int i = 0; i < num_nonzeros; i++) {
    int row = coo_rows[i];
    int pos = insert_pos[row]++;
    h_col_indices[pos] = coo_cols[i];
    h_values[pos] = coo_vals[i];
  }

  // Allocate device memory and upload
  CSRMatrix<Scalar> matrix;
  matrix.num_rows = num_rows;
  matrix.num_cols = num_cols;
  matrix.num_nonzeros = num_nonzeros;

  matrix.row_offsets = sycl::malloc_device<int>(num_rows + 1, queue);
  matrix.col_indices = sycl::malloc_device<int>(num_nonzeros, queue);
  matrix.values = sycl::malloc_device<Scalar>(num_nonzeros, queue);

  if (!matrix.row_offsets || !matrix.col_indices || !matrix.values) throw std::runtime_error("Device USM allocation failed");

  queue.memcpy(matrix.row_offsets, h_row_offsets, (num_rows + 1) * sizeof(int));
  queue.memcpy(matrix.col_indices, h_col_indices, num_nonzeros * sizeof(int));
  queue.memcpy(matrix.values, h_values, num_nonzeros * sizeof(Scalar));
  queue.wait();

  sycl::free(h_row_offsets, queue);
  sycl::free(h_col_indices, queue);
  sycl::free(h_values, queue);

  return matrix;
}

// Free CSR matrix USM allocations
template <class Scalar>
inline void free_csr_matrix(CSRMatrix<Scalar>& matrix, sycl::queue& queue)
{
  if (matrix.row_offsets) sycl::free(matrix.row_offsets, queue);
  if (matrix.col_indices) sycl::free(matrix.col_indices, queue);
  if (matrix.values) sycl::free(matrix.values, queue);
  matrix.row_offsets = nullptr;
  matrix.col_indices = nullptr;
  matrix.values = nullptr;
}

// CSR-based eigenvalue problem that inherits from StandardEVPBase
// Implements Y = A * X using a simple SYCL SpMM kernel
template <class T, unsigned int bs>
class CSREVP : public StandardEVPBase<T, bs> {
public:
  using Base = StandardEVPBase<T, bs>;

  CSREVP(sycl::queue queue, const std::string& matrix_file)
      : Base(queue, 0) // N will be set after loading
  {
    // Load the matrix from Matrix Market file
    matrix_ = load_matrix_market<T>(matrix_file, this->queue);

    if (matrix_.num_rows != matrix_.num_cols) throw std::runtime_error("CSREVP requires a square matrix");

    // Set the matrix dimension in the base class and reinitialize Vtemp
    this->N = matrix_.num_rows;
    this->Vtemp.emplace(this->create_multivector(this->N, bs));

    // Create cublas handle
    cusparseCreate(&handle);
    cusparseCreateCsr(&cumat, matrix_.num_rows, matrix_.num_rows, matrix_.num_nonzeros, matrix_.row_offsets, matrix_.col_indices, matrix_.values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    std::size_t bufferSize;
    T alpha = 1.;
    T beta = 0.;

    auto X = this->create_multivector(this->N, bs);
    auto Y = this->create_multivector(this->N, bs);
    this->queue.wait();

    cusparseDnMatDescr_t matX;
    cusparseDnMatDescr_t matY;
    cusparseCreateDnMat(&matX, this->N, bs, bs, X.block_view(0).data, CUDA_R_64F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matY, this->N, bs, bs, Y.block_view(0).data, CUDA_R_64F, CUSPARSE_ORDER_ROW);
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cumat, matX, &beta, matY, CUDA_R_64F,
                            CUSPARSE_SPMM_CSR_ALG2, &bufferSize);
    buf = sycl::malloc_device(bufferSize, this->queue);
    cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cumat, matX, &beta, matY, CUDA_R_64F,
                            CUSPARSE_SPMM_CSR_ALG2, buf);
    this->queue.wait();
  }

  ~CSREVP() { free_csr_matrix(matrix_, this->queue); }

  void apply(typename Base::BlockView X, typename Base::BlockView Y)
  {
    // Compute Y = A * X where A is sparse (CSR) and X is a tall-skinny matrix
    // X and Y are stored row-major with bs columns per row
    static auto* ev = trl::sycl::SyclProfiler::get().registerOrGetEvent(trl::sycl::SyclProfiler::get().registerOrGetFamily("CSREVP"), "apply");

#if 1
    const auto ncus = this->queue.get_device().template get_info<::sycl::info::device::max_compute_units>();
    const auto global_size = 128 * bs * ncus;

    T* X_data = X.data;
    T* Y_data = Y.data;
    const int* row_offsets = matrix_.row_offsets;
    const int* col_indices = matrix_.col_indices;
    const T* values = matrix_.values;
    const std::size_t num_rows = matrix_.num_rows;

    auto e = this->queue.parallel_for(sycl::range<1>(global_size), [=](sycl::id<1> idx) {
      const std::size_t start = idx[0];

      for (std::size_t row = start; row < num_rows; row += global_size) {
        const int row_start = row_offsets[row];
        const int row_end = row_offsets[row + 1];

        // Initialize output row to zero
        for (std::size_t j = 0; j < bs; ++j) Y_data[row * bs + j] = T(0);

        // Accumulate contributions from all nonzeros in this row
        for (int k = row_start; k < row_end; ++k) {
          const std::size_t col = col_indices[k];
          const T val = static_cast<T>(values[k]);

          for (std::size_t j = 0; j < bs; ++j) Y_data[row * bs + j] += val * X_data[col * bs + j];
        }
      }
    });
#else
    const std::size_t num_rows = matrix_.num_rows;
    T* X_data = X.data;
    T* Y_data = Y.data;

    auto e = this->queue.submit([=, handle = handle, cumat = cumat, buf = buf](auto& cgh) {
      T alpha = 1.;
      T beta = 0.;

      cgh.AdaptiveCpp_enqueue_custom_operation([=](auto& interop_handle) {
        auto stream = interop_handle.template get_native_queue<sycl::backend::cuda>();
        cusparseSetStream(handle, stream);

        cusparseDnMatDescr_t matX;
        cusparseDnMatDescr_t matY;

        cusparseCreateDnMat(&matX, num_rows, bs, bs, X_data, CUDA_R_64F, CUSPARSE_ORDER_ROW);
        cusparseCreateDnMat(&matY, num_rows, bs, bs, Y_data, CUDA_R_64F, CUSPARSE_ORDER_ROW);
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cumat, matX, &beta, matY, CUDA_R_64F,
                     CUSPARSE_SPMM_CSR_ALG2, buf);
      });
    });

#endif
    trl::sycl::SyclProfiler::get().pushEvent(ev, e);
  }

private:
  CSRMatrix<T> matrix_;

  cusparseHandle_t handle;
  cusparseSpMatDescr_t cumat;
  void* buf = nullptr;
};
