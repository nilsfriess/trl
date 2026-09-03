#pragma once

#include "trl/helpers.hh"
#include <algorithm>
#include <array>
#include <cassert>
#include <span>
#include <vector>

#include <sycl/sycl.hpp>

namespace trl::Sycl {
/** @brief SYCL matrix block view backed by device USM.
 *
 *  Backend specifics:
 *  - Holds a queue pointer used for memcpy/memset and kernel submission.
 *  - The pointed-to block lives in device memory (see BlockMatrix, which
 *    allocates with sycl::malloc_device), so it must never be dereferenced on
 *    the host. Every element-wise operation is therefore a kernel; host access
 *    goes through Backend::host_block, which stages the block via memcpy.
 *  - Methods do not wait; like the rest of the backend they assume an in-order
 *    queue for implicit dependency ordering.
 */
template <class T, unsigned int bs>
class MatrixBlockView {
public:
  using EntryType = T;
  static constexpr unsigned int rows = bs;
  static constexpr unsigned int cols = bs;

  MatrixBlockView(sycl::queue* queue, T* data)
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
    T* dest_ptr = data;
    const T* src_ptr = source.data;

    // Transpose into a private buffer before storing, so that the block may
    // alias its own source.
    queue->single_task([=] {
      T tmp[bs * bs];
      for (std::size_t i = 0; i < bs; ++i)
        for (std::size_t j = 0; j < bs; ++j) tmp[i * bs + j] = src_ptr[j * bs + i];

      for (std::size_t k = 0; k < bs * bs; ++k) dest_ptr[k] = tmp[k];
    });
  }

  void set_zero() { queue->memset(data, 0, bs * bs * sizeof(T)); }

  void set_diagonal(std::span<T> values)
  {
    assert(values.size() >= bs);

    // `values` is host memory, so it cannot be read from the kernel. Stage it
    // into the kernel functor instead; bs entries is small enough to pass as a
    // kernel argument, which avoids a scratch allocation and a second copy.
    std::array<T, bs> diag{};
    std::copy_n(values.begin(), bs, diag.begin());

    T* dest_ptr = data;
    queue->single_task([=] {
      for (std::size_t i = 0; i < bs; ++i) dest_ptr[i * bs + i] = diag[i];
    });
  }

  void mult(MatrixBlockView B, MatrixBlockView C)
  {
    const T* a_ptr = data;
    const T* b_ptr = B.data;
    T* c_ptr = C.data;

    // C = this * B (matrix-matrix multiplication). Accumulate into a private
    // buffer before storing, so that C may alias either input.
    queue->single_task([=] {
      T tmp[bs * bs];
      for (std::size_t i = 0; i < bs; ++i) {
        for (std::size_t j = 0; j < bs; ++j) {
          T sum{0};
          for (std::size_t k = 0; k < bs; ++k) sum += a_ptr[i * bs + k] * b_ptr[k * bs + j];
          tmp[i * bs + j] = sum;
        }
      }

      for (std::size_t k = 0; k < bs * bs; ++k) c_ptr[k] = tmp[k];
    });
  }

  T* data;

private:
  sycl::queue* queue;
};

} // namespace trl::Sycl
