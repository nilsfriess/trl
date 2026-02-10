#pragma once

#include "blockmatrix.hh"
#include "util.hh"

#include <algorithm>
#include <array>
#include <cstddef>

namespace trl::openmp {
template <class ScalarT, unsigned int block_size>
class BlockView {
public:
  using EntryType = ScalarT;
  using MatrixBlockView = typename BlockMatrix<ScalarT, block_size>::BlockView;

  void set_zero() { std::fill_n(data_, n_ * block_size, 0); }

  std::size_t rows() const { return n_; }
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

  void copy_from(BlockView other)
  {
#pragma omp parallel for
    for (std::size_t i = 0; i < n_ * block_size; ++i) data_[i] = other.data_[i];
  }

  BlockView& operator-=(BlockView other)
  {
#pragma omp parallel for
    for (std::size_t i = 0; i < n_ * block_size; ++i) data_[i] -= other.data_[i];
    return *this;
  }

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
