#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <sycl/sycl.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>

#include "trl/common.hh"
#include "trl/sycl/backend.hh"

// CSR operator implementing Y = A * X with a simple SYCL SpMM kernel.
// SYCL twin of examples/openmp/csrevp.hh: the matrix is loaded with Eigen and,
// as there, the stored triangle is interpreted as the lower half of a symmetric
// matrix. Only the kernel differs; the CSR arrays are mirrored into USM because
// the kernel cannot reach Eigen's host storage.
template <class T, unsigned int bs>
class CSREVP : public trl::EuclideanDot<trl::Sycl::Backend<T, bs>> {
public:
  using BlockView = typename trl::Sycl::Backend<T, bs>::Multivector::BlockView;

  CSREVP(sycl::queue queue, const std::string& matrix_file)
      : queue(queue)
  {
    Eigen::SparseMatrix<T> loaded;
    Eigen::loadMarket(loaded, matrix_file);

    if (loaded.rows() != loaded.cols()) throw std::runtime_error("CSREVP requires a square matrix");

    Eigen::SparseMatrix<T, Eigen::RowMajor> A = loaded.template selfadjointView<Eigen::Lower>();
    A.makeCompressed();

    n_ = static_cast<std::size_t>(A.rows());
    nnz_ = static_cast<std::size_t>(A.nonZeros());

    row_offsets = sycl::malloc_shared<int>(n_ + 1, queue);
    col_indices = sycl::malloc_shared<int>(nnz_, queue);
    values = sycl::malloc_shared<T>(nnz_, queue);
    if (!row_offsets || !col_indices || !values) throw std::runtime_error("USM allocation failed");

    std::copy_n(A.outerIndexPtr(), n_ + 1, row_offsets);
    std::copy_n(A.innerIndexPtr(), nnz_, col_indices);
    std::copy_n(A.valuePtr(), nnz_, values);
    queue.wait();
  }

  ~CSREVP()
  {
    sycl::free(row_offsets, queue);
    sycl::free(col_indices, queue);
    sycl::free(values, queue);
  }

  CSREVP(const CSREVP&) = delete;
  CSREVP& operator=(const CSREVP&) = delete;

  void apply(BlockView X, BlockView Y)
  {
    // Hoisted into locals: a device lambda cannot capture `this`.
    const int* row_offsets_ptr = row_offsets;
    const int* col_indices_ptr = col_indices;
    const T* values_ptr = values;
    const T* X_data = X.data;
    T* Y_data = Y.data;

    queue.parallel_for(sycl::range<1>(n_), [=](sycl::id<1> idx) {
      const std::size_t row = idx[0];
      const int row_start = row_offsets_ptr[row];
      const int row_end = row_offsets_ptr[row + 1];

      for (unsigned int i = 0; i < bs; ++i) Y_data[row * bs + i] = T(0);

      for (int k = row_start; k < row_end; ++k) {
        const std::size_t col = static_cast<std::size_t>(col_indices_ptr[k]);
        const T val = values_ptr[k];

        for (unsigned int i = 0; i < bs; ++i) Y_data[row * bs + i] += val * X_data[col * bs + i];
      }
    });
  }

  std::size_t size() const { return n_; }

private:
  sycl::queue queue;
  std::size_t n_{};
  std::size_t nnz_{};
  int* row_offsets = nullptr;
  int* col_indices = nullptr;
  T* values = nullptr;
};
