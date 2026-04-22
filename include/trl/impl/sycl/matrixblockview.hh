#pragma once

#include <cassert>
#include <span>

#include <sycl/sycl.hpp>

namespace trl::sycl {
/** @brief SYCL matrix block view backed by USM shared memory.
 *
 *  All operations are submitted as device kernels; no host access occurs
 *  inside the methods. Assumes an in-order queue for implicit dependency ordering.
 */
template <class T, unsigned int bs>
class MatrixBlockView {
public:
  using EntryType = T;
  static constexpr unsigned int rows = bs;
  static constexpr unsigned int cols = bs;

  MatrixBlockView(::sycl::queue* queue, T* data, T* diag_scratch)
      : data(data)
      , queue(queue)
      , diag_scratch(diag_scratch)
  {
  }

  MatrixBlockView(const MatrixBlockView&) = default;
  MatrixBlockView& operator=(const MatrixBlockView&) = default;
  MatrixBlockView(MatrixBlockView&&) = default;
  MatrixBlockView& operator=(MatrixBlockView&&) = default;
  ~MatrixBlockView() = default;

  void copy_from(const MatrixBlockView& source) { queue->memcpy(data, source.data, bs * bs * sizeof(T)); }

  void copy_from_transpose(const MatrixBlockView& source)
  {
    T* dest_ptr = data;
    T* src_ptr = source.data;
    queue->single_task([=]() {
      for (std::size_t i = 0; i < bs; ++i)
        for (std::size_t j = 0; j < bs; ++j) dest_ptr[i * bs + j] = src_ptr[j * bs + i];
    });
  }

  void set_zero() { queue->memset(data, 0, bs * bs * sizeof(T)); }

  void set_diagonal(std::span<T> values)
  {
    queue->memcpy(diag_scratch, values.data(), bs * sizeof(T));
    T* dest_ptr = data;
    T* scratch = diag_scratch;
    queue->single_task([=]() {
      for (std::size_t i = 0; i < bs; ++i) dest_ptr[i * bs + i] = scratch[i];
    });
  }

  void mult(MatrixBlockView B, MatrixBlockView C)
  {
    T* a = data;
    T* b = B.data;
    T* c = C.data;
    queue->single_task([=]() {
      for (std::size_t i = 0; i < bs; ++i)
        for (std::size_t j = 0; j < bs; ++j) {
          T sum = 0;
          for (std::size_t k = 0; k < bs; ++k) sum += a[i * bs + k] * b[k * bs + j];
          c[i * bs + j] = sum;
        }
    });
  }

  // Host-side element access; call queue->wait() before using after device ops.
  T& operator()(std::size_t row, std::size_t col)
  {
    assert(row < bs);
    assert(col < bs);
    return data[row * bs + col];
  }

  T* data;

private:
  ::sycl::queue* queue;
  T* diag_scratch;
};

} // namespace trl::sycl
