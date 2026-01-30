#pragma once

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "evp_base.hh"

// CSR (Compressed Sparse Row) matrix storage
struct CSRMatrix {
  int num_rows;
  int num_cols;
  int num_nonzeros;
  int* row_offsets; // size: num_rows + 1
  int* col_indices; // size: num_nonzeros
  double* values;   // size: num_nonzeros
};

// Read a Matrix Market file and assemble into CSR format using USM
inline CSRMatrix load_matrix_market(const std::string& filepath, sycl::queue& queue)
{
  std::ifstream file(filepath);
  if (!file) throw std::runtime_error("Error: cannot open file " + filepath);

  // Skip comments beginning with '%'
  while (file.peek() == '%') file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  // Read matrix dimensions and number of nonzeros
  int num_rows, num_cols, num_nonzeros;
  file >> num_rows >> num_cols >> num_nonzeros;

  // Allocate CSR arrays in unified shared memory
  CSRMatrix matrix;
  matrix.num_rows = num_rows;
  matrix.num_cols = num_cols;
  matrix.num_nonzeros = num_nonzeros;

  matrix.row_offsets = sycl::malloc_shared<int>(num_rows + 1, queue);
  matrix.col_indices = sycl::malloc_shared<int>(num_nonzeros, queue);
  matrix.values = sycl::malloc_shared<double>(num_nonzeros, queue);

  if (!matrix.row_offsets || !matrix.col_indices || !matrix.values) throw std::runtime_error("USM allocation failed");

  // Initialize row_offsets with zeros for counting
  for (int i = 0; i <= num_rows; i++) matrix.row_offsets[i] = 0;

  // First pass: read COO data and count entries per row
  std::vector<int> coo_rows(num_nonzeros);
  std::vector<int> coo_cols(num_nonzeros);
  std::vector<double> coo_vals(num_nonzeros);

  for (int i = 0; i < num_nonzeros; i++) {
    file >> coo_rows[i] >> coo_cols[i] >> coo_vals[i];
    coo_rows[i]--; // Convert 1-based to 0-based indexing
    coo_cols[i]--;
    matrix.row_offsets[coo_rows[i] + 1]++;
  }

  // Convert counts to prefix sum (row_offsets)
  for (int i = 1; i <= num_rows; i++) matrix.row_offsets[i] += matrix.row_offsets[i - 1];

  // Second pass: fill col_indices and values
  std::vector<int> insert_pos(matrix.row_offsets, matrix.row_offsets + num_rows);

  for (int i = 0; i < num_nonzeros; i++) {
    int row = coo_rows[i];
    int pos = insert_pos[row]++;
    matrix.col_indices[pos] = coo_cols[i];
    matrix.values[pos] = coo_vals[i];
  }

  return matrix;
}

// Free CSR matrix USM allocations
inline void free_csr_matrix(CSRMatrix& matrix, sycl::queue& queue)
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
    matrix_ = load_matrix_market(matrix_file, this->queue);

    if (matrix_.num_rows != matrix_.num_cols) throw std::runtime_error("CSREVP requires a square matrix");

    // Set the matrix dimension in the base class and reinitialize Vtemp
    this->N = matrix_.num_rows;
    this->Vtemp.emplace(this->create_multivector(this->N, bs));
  }

  ~CSREVP() { free_csr_matrix(matrix_, this->queue); }

  void apply(typename Base::BlockView X, typename Base::BlockView Y)
  {
    // Compute Y = A * X where A is sparse (CSR) and X is a tall-skinny matrix
    // X and Y are stored row-major with bs columns per row

    T* X_data = X.data;
    T* Y_data = Y.data;
    const int* row_offsets = matrix_.row_offsets;
    const int* col_indices = matrix_.col_indices;
    const double* values = matrix_.values;
    const std::size_t num_rows = matrix_.num_rows;

    this->queue.parallel_for(sycl::range<1>(num_rows), [=](sycl::id<1> idx) {
      const std::size_t row = idx[0];
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
    });
  }

private:
  CSRMatrix matrix_;
};
