#pragma once

#include <cassert>
#include <cstddef>

#include <sycl/sycl.hpp>

#include <oneapi/math.hpp>

#include "matrixblockview.hh"

namespace trl {
using namespace oneapi::math;
using oneapi::math::blas::row_major::axpy;
using oneapi::math::blas::row_major::gemm;

template <class T, std::size_t cols_>
class BlockView {
public:
  using EntryType = T;
  using MatrixBlockView = MatrixBlockView<T, cols_>;

  BlockView(sycl::queue* queue, T* data, std::size_t rows)
      : data(data)
      , queue(queue)
      , rows_(rows)
  {
  }

  // Default copy operations (copying a view is cheap)
  BlockView(const BlockView&) = default;
  BlockView& operator=(const BlockView&) = default;

  // Default move operations
  BlockView(BlockView&&) = default;
  BlockView& operator=(BlockView&&) = default;

  // Default destructor (view doesn't own data)
  ~BlockView() = default;

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  /** @brief Copy data from another block view
   *
   *  Copies the data from source into this view. Both views must have
   *  the same dimensions.
   */
  void copy_from(const BlockView& source)
  {
    assert(rows_ == source.rows_);
    queue->memcpy(data, source.data, rows_ * cols_ * sizeof(T)).wait();
  }

  void dot(BlockView B, MatrixBlockView C) { return gemm(*queue, transpose::trans, transpose::nontrans, cols_, cols_, rows_, 1., data, cols_, B.data, cols_, 0., C.data, cols_).wait(); }

  void mult(MatrixBlockView B, BlockView C) { return gemm(*queue, transpose::nontrans, transpose::nontrans, rows_, cols_, cols_, 1., data, cols_, B.data, cols_, 0., C.data, cols_).wait(); }

  void mult_add(MatrixBlockView B, BlockView C) { return gemm(*queue, transpose::nontrans, transpose::nontrans, rows_, cols_, cols_, 1., data, cols_, B.data, cols_, 1., C.data, cols_).wait(); }

  void mult_transpose(MatrixBlockView B, BlockView C) { return gemm(*queue, transpose::nontrans, transpose::trans, rows_, cols_, cols_, 1., data, cols_, B.data, cols_, 0., C.data, cols_).wait(); }

  BlockView& operator-=(BlockView B)
  {
    axpy(*queue, rows_ * cols_, -1., B.data, 1, data, 1).wait();
    return *this;
  }

  void subtract_product(BlockView B, MatrixBlockView C) { gemm(*queue, transpose::nontrans, transpose::nontrans, rows_, cols_, cols_, -1., B.data, cols_, C.data, cols_, 1., data, cols_).wait(); }

  T norm() const
  {
    T result = 0;
    oneapi::math::blas::row_major::nrm2(*queue, rows_ * cols_, data, 1, &result).wait();
    return result;
  }

  void set_zero() { queue->memset(data, 0, rows_ * cols_ * sizeof(T)).wait(); }

  T& operator()(std::size_t row, std::size_t col) { return data[row * cols_ + col]; }

  T* data;

private:
  sycl::queue* queue;
  const std::size_t rows_;
};
} // namespace trl
