#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "trl/impl/openmp/evp_base.hh"
#include "trl/impl/openmp/util.hh"

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
inline CSRMatrix load_matrix_market(const std::string& filepath)
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
  std::vector<double> coo_vals;
  coo_rows.reserve(is_symmetric ? 2 * num_entries_in_file : num_entries_in_file);
  coo_cols.reserve(is_symmetric ? 2 * num_entries_in_file : num_entries_in_file);
  coo_vals.reserve(is_symmetric ? 2 * num_entries_in_file : num_entries_in_file);

  for (int i = 0; i < num_entries_in_file; i++) {
    int row, col;
    double val;
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

  // Allocate CSR arrays in unified shared memory
  CSRMatrix matrix;
  matrix.num_rows = num_rows;
  matrix.num_cols = num_cols;
  matrix.num_nonzeros = num_nonzeros;

  matrix.row_offsets = (int*)std::aligned_alloc(64, (num_rows + 1) * sizeof(int));
  matrix.col_indices = (int*)std::aligned_alloc(64, num_nonzeros * sizeof(int));
  matrix.values = (double*)std::aligned_alloc(64, num_nonzeros * sizeof(double));

  if (!matrix.row_offsets || !matrix.col_indices || !matrix.values) throw std::runtime_error("USM allocation failed");

  // Initialize row_offsets with zeros for counting
  for (int i = 0; i <= num_rows; i++) matrix.row_offsets[i] = 0;

  // Count entries per row
  for (int i = 0; i < num_nonzeros; i++) matrix.row_offsets[coo_rows[i] + 1]++;

  // Convert counts to prefix sum (row_offsets)
  for (int i = 1; i <= num_rows; i++) matrix.row_offsets[i] += matrix.row_offsets[i - 1];

  // Fill col_indices and values
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
inline void free_csr_matrix(CSRMatrix& matrix)
{
  if (matrix.row_offsets) std::free(matrix.row_offsets);
  if (matrix.col_indices) std::free(matrix.col_indices);
  if (matrix.values) std::free(matrix.values);
  matrix.row_offsets = nullptr;
  matrix.col_indices = nullptr;
  matrix.values = nullptr;
}

// CSR-based eigenvalue problem that inherits from StandardEVPBase
// Implements Y = A * X using a simple SYCL SpMM kernel
template <class T, unsigned int bs>
class CSREVP : public trl::openmp::EVPBase<T, bs> {
public:
  using Base = trl::openmp::EVPBase<T, bs>;

  CSREVP(const std::string& matrix_file)
      : Base(0) // N will be set after loading
  {
    // Load the matrix from Matrix Market file
    matrix_ = load_matrix_market(matrix_file);

    if (matrix_.num_rows != matrix_.num_cols) throw std::runtime_error("CSREVP requires a square matrix");

    // Set the matrix dimension in the base class and reinitialize Vtemp
    this->N = matrix_.num_rows;
    this->Vtemp.emplace(this->create_multivector(this->N, bs));
  }

  ~CSREVP() { free_csr_matrix(matrix_); }

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

#pragma omp parallel for
    for (std::size_t row = 0; row < num_rows; ++row) {
      const int row_start = row_offsets[row];
      const int row_end = row_offsets[row + 1];

      alignas(64) T y_local[bs];
#pragma omp simd
      for (std::size_t j = 0; j < bs; ++j) y_local[j] = T(0);

      for (int k = row_start; k < row_end; ++k) {
        const std::size_t col = static_cast<std::size_t>(col_indices[k]);
        const T val = static_cast<T>(values[k]);

#pragma omp simd
        for (std::size_t j = 0; j < bs; ++j) y_local[j] += val * X_data[col * bs + j];
      }

#pragma omp simd
      for (std::size_t j = 0; j < bs; ++j) Y_data[row * bs + j] = y_local[j];
    }
  }

private:
  CSRMatrix matrix_;
};
