#pragma once

#include <cassert>
#include <span>
#include <vector>

#include <sycl/sycl.hpp>

namespace trl::sycl {
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

  /** @brief Copy data from another matrix block view
   *
   *  Copies the data from source into this view. Both views must have
   *  the same dimensions (bs × bs).
   */
  void copy_from(const MatrixBlockView& source) { queue->memcpy(data, source.data, bs * bs * sizeof(T)); }

  /** @brief Copy and transpose data from another matrix block view
   *
   *  Copies the transposed data from source into this view.
   *  This[i,j] = source[j,i] for all i,j.
   */
  void copy_from_transpose(const MatrixBlockView& source)
  {
    T* dest_ptr = data;
    T* src_ptr = source.data;
    queue->submit([&](::sycl::handler& h) {
      h.parallel_for(::sycl::range<2>(bs, bs), [=](::sycl::id<2> idx) {
        const auto i = idx[0];
        const auto j = idx[1];
        dest_ptr[i * bs + j] = src_ptr[j * bs + i];
      });
    });
  }

  /** @brief Set all elements of the block to zero */
  void set_zero() { queue->memset(data, 0, bs * bs * sizeof(T)); }

  /** @brief Set the diagonal elements of the block
   *
   *  Sets diagonal elements data[i,i] = values[i] for i in [0, bs).
   *  Off-diagonal elements are unchanged.
   *
   *  @param values Vector of size bs containing the diagonal values
   */
  void set_diagonal(std::span<T, bs> values)
  {
    T* dest_ptr = data;

    queue->submit([&](::sycl::handler& h) {
      h.parallel_for(::sycl::range<1>(bs), [=](::sycl::id<1> idx) {
        const auto i = idx[0];
        dest_ptr[i * bs + i] = values[i];
      });
    });
  }

  /** @brief Multiply this block matrix by a small vector
   *
   *  Computes result = this * vec, where:
   *  - this is a bs × bs matrix
   *  - vec is a vector of size bs
   *  - result is a vector of size bs
   *
   *  @param vec Input vector of size bs
   *  @param result Output vector of size bs (will be overwritten)
   */
  void mult(const std::vector<T>& vec, std::vector<T>& result) const { assert(false && "not implemented"); }

  void mult(MatrixBlockView B, MatrixBlockView C) { assert(false && "not implemented"); }

  /** @brief Element access operator for reading/writing matrix elements
   *
   *  Provides direct access to matrix element at (row, col).
   *  Note: This requires synchronization if used on device memory.
   *
   *  @param row Row index (0-based)
   *  @param col Column index (0-based)
   *  @return Reference to the element at (row, col)
   */
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
