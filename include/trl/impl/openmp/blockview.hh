#pragma once

#include "blockmatrix.hh"
#include "util.hh"

#include <algorithm>
#include <array>
#include <cstddef>

namespace trl::openmp {
  /** @brief OpenMP block view backed by aligned host memory.
   *
   *  Backend specifics:
   *  - Data is stored in row-major order with 64-byte alignment.
   *  - Operations use OpenMP parallel loops and SIMD pragmas.
   *  - The view is lightweight (std::span-like); copying the view is cheap
   *    but does not copy the underlying data.
   *  - dot() uses thread-local buffers combined in a critical section.
   */
template <class ScalarT, unsigned int block_size>
class BlockView {
public:
  constexpr static auto blocksize = block_size;
  using EntryType = ScalarT;
  using MatrixBlockView = typename BlockMatrix<ScalarT, block_size>::BlockView;

  /** @brief Sets all entries to zero.
   *
   *  Computes \f$ X_{ij} = 0 \f$ for all \f$ i, j \f$.
   */
  void set_zero() { std::fill_n(data_, n_ * block_size, 0); }

  /** @brief Returns the number of rows in the block. */
  std::size_t rows() const { return n_; }

  /** @brief Returns the block size (number of columns). */
  constexpr std::size_t cols() const { return block_size; }

  BlockView(ScalarT* data, std::size_t n)
      : data_(data)
      , n_(n)
  {
  }

  BlockView(const BlockView&) = default;
  BlockView& operator=(const BlockView&) = default;
  BlockView(BlockView&&) = default;
  BlockView& operator=(BlockView&&) = default;

  /** @brief Copies data from another block view.
   *
   *  Computes the element-wise assignment \f$ X = Y \f$.
   */
  void copy_from(BlockView other)
  {
#pragma omp parallel for
    for (std::size_t i = 0; i < n_ * block_size; ++i) data_[i] = other.data_[i];
  }

  /** @brief Subtracts another block view element-wise.
   *
   *  Computes \f$ X = X - Y \f$ where X is this block view.
   */
  BlockView& operator-=(BlockView other)
  {
#pragma omp parallel for
    for (std::size_t i = 0; i < n_ * block_size; ++i) data_[i] -= other.data_[i];
    return *this;
  }

  /** @brief Computes matrix-block product and adds to result.
   *
   *  Computes \f$ Y = Y + X W \f$ where X is this block view,
   *  W is a small square block matrix, and Y is the output.
   */
  void mult_add(MatrixBlockView W, BlockView other)
  {
    // other += this * W
#pragma omp parallel for
    for (std::size_t k = 0; k < n_; ++k) {
      alignas(64) std::array<ScalarT, block_size> tmp{};
      for (std::size_t i = 0; i < block_size; ++i) {
        ScalarT sum = 0;
#pragma omp simd reduction(+ : sum)
        for (std::size_t j = 0; j < block_size; ++j) sum += data_[k * block_size + j] * W.data_[j * block_size + i];
        tmp[i] = sum;
      }

#pragma omp simd
      for (std::size_t i = 0; i < block_size; ++i) other.data_[k * block_size + i] += tmp[i];
    }
  }

  /** @brief Computes matrix-block product.
   *
   *  Computes \f$ Y = X W \f$ where X is this block view
   *  and W is a small square block matrix.
   */
  void mult(MatrixBlockView W, BlockView other)
  {
    other.set_zero();
#pragma omp parallel for
    for (std::size_t k = 0; k < n_; ++k) {
      alignas(64) std::array<ScalarT, block_size> tmp{};
      for (std::size_t i = 0; i < block_size; ++i) {
        ScalarT sum = 0;
#pragma omp simd reduction(+ : sum)
        for (std::size_t j = 0; j < block_size; ++j) sum += data_[k * block_size + j] * W.data_[j * block_size + i];
        tmp[i] = sum;
      }

#pragma omp simd
      for (std::size_t i = 0; i < block_size; ++i) other.data_[k * block_size + i] = tmp[i];
    }
  }

  /** @brief Computes matrix-block product with transposed block matrix.
   *
   *  Computes \f$ Y = X W^T \f$ where X is this block view
   *  and \f$ W^T \f$ is the transpose of the block matrix.
   */
  void mult_transpose(MatrixBlockView W, BlockView other)
  {
    other.set_zero();
#pragma omp parallel for
    for (std::size_t k = 0; k < n_; ++k) {
      alignas(64) std::array<ScalarT, block_size> tmp{};
      for (std::size_t i = 0; i < block_size; ++i) {
        ScalarT sum = 0;
#pragma omp simd reduction(+ : sum)
        for (std::size_t j = 0; j < block_size; ++j) sum += data_[k * block_size + j] * W.data_[i * block_size + j];
        tmp[i] = sum;
      }

#pragma omp simd
      for (std::size_t i = 0; i < block_size; ++i) other.data_[k * block_size + i] = tmp[i];
    }
  }

  /** @brief Computes block inner product.
   *
   *  Computes \f$ R = X^T Y \f$ where X is this block view,
   *  Y is the other block view, and R is a small square block matrix.
   *
   *  @par Implementation:
   *  Uses thread-local buffers to accumulate partial results, which are
   *  combined in a critical section to avoid race conditions.
   */
  void dot(BlockView other, MatrixBlockView R)
  {
    R.set_zero();

#pragma omp parallel
    {
      alignas(64) std::array<ScalarT, block_size * block_size> R_temp{};
#pragma omp for
      for (std::size_t k = 0; k < n_ - 3; k += 4) {
        for (std::size_t i = 0; i < block_size; ++i)
#pragma omp simd
          for (std::size_t j = 0; j < block_size; ++j) {
            R_temp[i * block_size + j] += data_[(k + 0) * block_size + i] * other.data_[(k + 0) * block_size + j] + data_[(k + 1) * block_size + i] * other.data_[(k + 1) * block_size + j] +
                                          data_[(k + 2) * block_size + i] * other.data_[(k + 2) * block_size + j] + data_[(k + 3) * block_size + i] * other.data_[(k + 3) * block_size + j];
          }
      }

#pragma omp critical
      {
        for (std::size_t i = 0; i < block_size * block_size; ++i) R.data_[i] += R_temp[i];
      }
    }

    // Sequential epilogue for remaining rows
    for (std::size_t k = (n_ / 4) * 4; k < n_; ++k) {
      for (std::size_t i = 0; i < block_size; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < block_size; ++j) R.data_[i * block_size + j] += data_[k * block_size + i] * other.data_[k * block_size + j];
    }
  }

  /** @brief Subtracts a matrix-block product from this block view.
   *
   *  Computes \f$ X = X - Y R \f$ where X is this block view,
   *  Y is another block view, and R is a small square block matrix.
   */
  void subtract_product(BlockView other, MatrixBlockView R)
  {
    // this -= other * R
#pragma omp parallel for
    for (std::size_t k = 0; k < n_; ++k) {
      for (std::size_t i = 0; i < block_size; ++i) {
        ScalarT sum = 0;
#pragma omp simd reduction(+ : sum)
        for (std::size_t j = 0; j < block_size; ++j) sum += other.data_[k * block_size + j] * R.data_[j * block_size + i];
        data_[k * block_size + i] -= sum;
      }
    }
  }

  ScalarT* data_;

private:
  std::size_t n_;
};
} // namespace trl::openmp
