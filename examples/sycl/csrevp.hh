#pragma once

#include "evp_base.hh"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <trl/impl/sycl/profiling.hh>
#include <vector>

// CSR (Compressed Sparse Row) matrix storage
template <class Scalar>
class CSRMatrix {
public:
  CSRMatrix(const std::string& filepath, sycl::queue& queue)
      : queue(queue)
  {
    load(filepath, queue);
  }

  CSRMatrix(const std::string& filepath_a, const std::string& filepath_b, Scalar shift, sycl::queue& queue)
      : queue(queue)
  {
    load_shifted(filepath_a, filepath_b, shift, queue);
  }

  ~CSRMatrix()
  {
    if (row_offsets_) sycl::free(row_offsets_, queue);
    if (col_indices_) sycl::free(col_indices_, queue);
    if (values_) sycl::free(values_, queue);
    row_offsets_ = nullptr;
    col_indices_ = nullptr;
    values_ = nullptr;
  }

  int rows() const { return num_rows_; }
  int cols() const { return num_cols_; }
  int nnz() const { return num_nonzeros_; }

  const int* rowOffsets() const { return row_offsets_; }
  const int* colIndices() const { return col_indices_; }
  const Scalar* values() const { return values_; }

private:
  void load(const std::string& filepath, sycl::queue& queue)
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
    num_rows_ = num_rows;
    num_cols_ = num_cols;
    num_nonzeros_ = num_nonzeros;

    row_offsets_ = sycl::malloc_device<int>(num_rows + 1, queue);
    col_indices_ = sycl::malloc_device<int>(num_nonzeros, queue);
    values_ = sycl::malloc_device<Scalar>(num_nonzeros, queue);

    if (!row_offsets_ || !col_indices_ || !values_) throw std::runtime_error("Device USM allocation failed");

    queue.memcpy(row_offsets_, h_row_offsets, (num_rows + 1) * sizeof(int));
    queue.memcpy(col_indices_, h_col_indices, num_nonzeros * sizeof(int));
    queue.memcpy(values_, h_values, num_nonzeros * sizeof(Scalar));
    queue.wait();

    sycl::free(h_row_offsets, queue);
    sycl::free(h_col_indices, queue);
    sycl::free(h_values, queue);
  }

  void load_shifted(const std::string& filepath_a, const std::string& filepath_b, Scalar shift, sycl::queue& queue)
  {
    // --- helper lambda: parse a Matrix Market file into sorted COO ---
    auto parse_coo = [](const std::string& filepath, int& out_rows, int& out_cols, std::vector<int>& rows, std::vector<int>& cols,
                        std::vector<Scalar>& vals) {
      std::ifstream file(filepath);
      if (!file) throw std::runtime_error("Error: cannot open file " + filepath);

      bool is_symmetric = false;
      std::string line;
      while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '%') {
          if (line.find("%%MatrixMarket") != std::string::npos || line.find("%%matrixmarket") != std::string::npos) {
            std::string lower = line;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower.find("symmetric") != std::string::npos) is_symmetric = true;
          }
          continue;
        }
        break;
      }

      int num_entries_in_file;
      std::istringstream iss(line);
      iss >> out_rows >> out_cols >> num_entries_in_file;

      rows.reserve(is_symmetric ? 2 * num_entries_in_file : num_entries_in_file);
      cols.reserve(is_symmetric ? 2 * num_entries_in_file : num_entries_in_file);
      vals.reserve(is_symmetric ? 2 * num_entries_in_file : num_entries_in_file);

      for (int i = 0; i < num_entries_in_file; i++) {
        int r, c;
        Scalar v;
        file >> r >> c >> v;
        r--;
        c--;
        rows.push_back(r);
        cols.push_back(c);
        vals.push_back(v);
        if (is_symmetric && r != c) {
          rows.push_back(c);
          cols.push_back(r);
          vals.push_back(v);
        }
      }

      // Sort by (row, col)
      const int n = static_cast<int>(rows.size());
      std::vector<int> idx(n);
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(), [&](int a, int b) { return rows[a] < rows[b] || (rows[a] == rows[b] && cols[a] < cols[b]); });
      std::vector<int> sr(n), sc(n);
      std::vector<Scalar> sv(n);
      for (int i = 0; i < n; i++) {
        sr[i] = rows[idx[i]];
        sc[i] = cols[idx[i]];
        sv[i] = vals[idx[i]];
      }
      rows = std::move(sr);
      cols = std::move(sc);
      vals = std::move(sv);
    };

    int num_rows_a, num_cols_a, num_rows_b, num_cols_b;
    std::vector<int> rows_a, cols_a, rows_b, cols_b;
    std::vector<Scalar> vals_a, vals_b;

    parse_coo(filepath_a, num_rows_a, num_cols_a, rows_a, cols_a, vals_a);
    parse_coo(filepath_b, num_rows_b, num_cols_b, rows_b, cols_b, vals_b);

    if (num_rows_a != num_rows_b || num_cols_a != num_cols_b) throw std::runtime_error("Matrix dimensions mismatch: cannot compute A - shift*B");

    const int num_rows = num_rows_a;
    const int num_cols = num_cols_a;

    // Merge the two sorted COO arrays: C = A - shift * B
    std::vector<int> coo_rows, coo_cols;
    std::vector<Scalar> coo_vals;
    coo_rows.reserve(rows_a.size() + rows_b.size());
    coo_cols.reserve(cols_a.size() + cols_b.size());
    coo_vals.reserve(vals_a.size() + vals_b.size());

    int ia = 0, ib = 0;
    const int na = static_cast<int>(rows_a.size());
    const int nb = static_cast<int>(rows_b.size());
    while (ia < na && ib < nb) {
      const bool a_first = rows_a[ia] < rows_b[ib] || (rows_a[ia] == rows_b[ib] && cols_a[ia] < cols_b[ib]);
      const bool b_first = rows_b[ib] < rows_a[ia] || (rows_b[ib] == rows_a[ia] && cols_b[ib] < cols_a[ia]);
      if (a_first) {
        coo_rows.push_back(rows_a[ia]);
        coo_cols.push_back(cols_a[ia]);
        coo_vals.push_back(vals_a[ia++]);
      }
      else if (b_first) {
        coo_rows.push_back(rows_b[ib]);
        coo_cols.push_back(cols_b[ib]);
        coo_vals.push_back(-shift * vals_b[ib++]);
      }
      else {
        // Same (row, col): a_ij - shift * b_ij
        coo_rows.push_back(rows_a[ia]);
        coo_cols.push_back(cols_a[ia]);
        coo_vals.push_back(vals_a[ia] - shift * vals_b[ib]);
        ia++;
        ib++;
      }
    }
    while (ia < na) {
      coo_rows.push_back(rows_a[ia]);
      coo_cols.push_back(cols_a[ia]);
      coo_vals.push_back(vals_a[ia++]);
    }
    while (ib < nb) {
      coo_rows.push_back(rows_b[ib]);
      coo_cols.push_back(cols_b[ib]);
      coo_vals.push_back(-shift * vals_b[ib++]);
    }

    const int num_nonzeros = static_cast<int>(coo_rows.size());

    // Build CSR and upload to device (same as load())
    int* h_row_offsets = sycl::malloc_host<int>(num_rows + 1, queue);
    int* h_col_indices = sycl::malloc_host<int>(num_nonzeros, queue);
    Scalar* h_values = sycl::malloc_host<Scalar>(num_nonzeros, queue);

    for (int i = 0; i <= num_rows; i++) h_row_offsets[i] = 0;
    for (int i = 0; i < num_nonzeros; i++) h_row_offsets[coo_rows[i] + 1]++;
    for (int i = 1; i <= num_rows; i++) h_row_offsets[i] += h_row_offsets[i - 1];

    std::vector<int> insert_pos(h_row_offsets, h_row_offsets + num_rows);
    for (int i = 0; i < num_nonzeros; i++) {
      int pos = insert_pos[coo_rows[i]]++;
      h_col_indices[pos] = coo_cols[i];
      h_values[pos] = coo_vals[i];
    }

    num_rows_ = num_rows;
    num_cols_ = num_cols;
    num_nonzeros_ = num_nonzeros;

    row_offsets_ = sycl::malloc_device<int>(num_rows + 1, queue);
    col_indices_ = sycl::malloc_device<int>(num_nonzeros, queue);
    values_ = sycl::malloc_device<Scalar>(num_nonzeros, queue);

    if (!row_offsets_ || !col_indices_ || !values_) throw std::runtime_error("Device USM allocation failed");

    queue.memcpy(row_offsets_, h_row_offsets, (num_rows + 1) * sizeof(int));
    queue.memcpy(col_indices_, h_col_indices, num_nonzeros * sizeof(int));
    queue.memcpy(values_, h_values, num_nonzeros * sizeof(Scalar));
    queue.wait();

    sycl::free(h_row_offsets, queue);
    sycl::free(h_col_indices, queue);
    sycl::free(h_values, queue);
  }

  sycl::queue queue;

  int num_rows_;
  int num_cols_;
  int num_nonzeros_;
  int* row_offsets_; // size: num_rows + 1
  int* col_indices_; // size: num_nonzeros
  Scalar* values_;   // size: num_nonzeros
};

// CSR-based eigenvalue problem that inherits from StandardEVPBase
// Implements Y = A * X using a simple SYCL SpMM kernel
template <class T, unsigned int bs>
class CSREVP : public StandardEVPBase<T, bs> {
public:
  using Base = StandardEVPBase<T, bs>;

  CSREVP(sycl::queue queue, const std::string& matrix_file)
      : Base(queue, 0)
      , matrix_(matrix_file, queue)
  {
    if (matrix_.rows() != matrix_.cols()) throw std::runtime_error("CSREVP requires a square matrix");

    // Set the matrix dimension in the base class and reinitialize Vtemp
    this->N = matrix_.rows();
    this->Vtemp.emplace(this->create_multivector(this->N, bs));
  }

  void apply(typename Base::BlockView X, typename Base::BlockView Y)
  {
    // Compute Y = A * X where A is sparse (CSR) and X is a tall-skinny matrix
    // X and Y are stored row-major with bs columns per row
    static auto* ev = trl::sycl::SyclProfiler::get().registerOrGetEvent(trl::sycl::SyclProfiler::get().registerOrGetFamily("CSREVP"), "apply");

    const auto ncus = this->queue.get_device().template get_info<::sycl::info::device::max_compute_units>();
    const auto global_size = 128 * bs * ncus;

    T* X_data = X.data;
    T* Y_data = Y.data;
    const int* row_offsets = matrix_.rowOffsets();
    const int* col_indices = matrix_.colIndices();
    const T* values = matrix_.values();
    const std::size_t num_rows = matrix_.rows();

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
    trl::sycl::SyclProfiler::get().pushEvent(ev, e);
  }

private:
  CSRMatrix<T> matrix_;
};
