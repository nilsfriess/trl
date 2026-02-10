#pragma once

#include <cassert>
#include <span>
#include <vector>

#include <sycl/sycl.hpp>

namespace trl::sycl {
/** @brief SYCL matrix block view backed by USM shared memory.
 *
 *  Backend specifics:
 *  - Holds a queue pointer used for memcpy/memset and synchronization.
 *  - Methods that read/write on the host call queue->wait().
 *  - Assumes an in-order queue for implicit dependency ordering.
 */
template <class T, unsigned int bs>
class MatrixBlockView {
public:
  using EntryType = T;
  static constexpr unsigned int rows = bs;
  static constexpr unsigned int cols = bs;

  MatrixBlockView(::sycl::queue* queue, T* data)
      : data(data)
      , queue(queue)
  {
  }

  // Default copy operations (copying a view is cheap, like std::span)
  MatrixBlockView(const MatrixBlockView&) = default;
  MatrixBlockView& operator=(const MatrixBlockView&) = default;

  // Default move operations
  MatrixBlockView(MatrixBlockView&&) = default;
  MatrixBlockView& operator=(MatrixBlockView&&) = default;

  // Default destructor (view doesn't own data)
  ~MatrixBlockView() = default;

  void copy_from(const MatrixBlockView& source) { queue->memcpy(data, source.data, bs * bs * sizeof(T)); }

  void copy_from_transpose(const MatrixBlockView& source)
  {
    queue->wait();
    T* dest_ptr = data;
    T* src_ptr = source.data;
    for (std::size_t i = 0; i < bs; ++i)
      for (std::size_t j = 0; j < bs; ++j) dest_ptr[i * bs + j] = src_ptr[j * bs + i];
  }

  void set_zero() { queue->memset(data, 0, bs * bs * sizeof(T)); }

  void set_diagonal(std::span<T> values)
  {
    queue->wait();
    T* dest_ptr = data;
    for (std::size_t i = 0; i < bs; ++i) dest_ptr[i * bs + i] = values[i];
  }

  void mult(const std::vector<T>& vec, std::vector<T>& result) const { assert(false && "not implemented"); }

  void mult(MatrixBlockView B, MatrixBlockView C)
  {
    queue->wait();

    // C = this * B (matrix-matrix multiplication)
    for (std::size_t i = 0; i < bs; ++i) {
      for (std::size_t j = 0; j < bs; ++j) {
        C(i, j) = 0;
        for (std::size_t k = 0; k < bs; ++k) C(i, j) += (*this)(i, k) * B(k, j);
      }
    }
  }

  T& operator()(std::size_t row, std::size_t col)
  {
    assert(row < bs);
    assert(col < bs);
    return data[row * bs + col];
  }

  T* data;

private:
  ::sycl::queue* queue;
};

} // namespace trl::sycl
