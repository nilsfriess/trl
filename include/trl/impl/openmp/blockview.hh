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

  void set_zero() { std::fill_n(data, n * block_size, 0); }

  std::size_t rows() const { return n; }
  constexpr std::size_t cols() const { return block_size; }

  BlockView(ScalarT* data, std::size_t n)
      : data(data)
      , n(n)
  {
  }

  BlockView(const BlockView&) = default;
  BlockView& operator=(const BlockView&) = default;
  BlockView(BlockView&&) = default;
  BlockView& operator=(BlockView&&) = default;

  void copy_from(BlockView other)
  {
#pragma omp parallel for
    for (std::size_t i = 0; i < n * block_size; ++i) data[i] = other.data[i];
  }

  BlockView& operator-=(BlockView other)
  {
#pragma omp parallel for
    for (std::size_t i = 0; i < n * block_size; ++i) data[i] -= other.data[i];
    return *this;
  }

  void mult_add(MatrixBlockView W, BlockView other)
  {
    // other += this * W
#pragma omp parallel for
    for (std::size_t k = 0; k < n; ++k) {
      alignas(64) std::array<ScalarT, block_size> tmp{};
      for (std::size_t i = 0; i < block_size; ++i) {
        ScalarT sum = 0;
#pragma omp simd reduction(+ : sum)
        for (std::size_t j = 0; j < block_size; ++j) sum += data[k * block_size + j] * W.data[j * block_size + i];
        tmp[i] = sum;
      }

#pragma omp simd
      for (std::size_t i = 0; i < block_size; ++i) other.data[k * block_size + i] += tmp[i];
    }
  }

  void mult(MatrixBlockView W, BlockView other)
  {
    other.set_zero();
#pragma omp parallel for
    for (std::size_t k = 0; k < n; ++k) {
      alignas(64) std::array<ScalarT, block_size> tmp{};
      for (std::size_t i = 0; i < block_size; ++i) {
        ScalarT sum = 0;
#pragma omp simd reduction(+ : sum)
        for (std::size_t j = 0; j < block_size; ++j) sum += data[k * block_size + j] * W.data[j * block_size + i];
        tmp[i] = sum;
      }

#pragma omp simd
      for (std::size_t i = 0; i < block_size; ++i) other.data[k * block_size + i] = tmp[i];
    }
  }

  void mult_transpose(MatrixBlockView W, BlockView other)
  {
    other.set_zero();
#pragma omp parallel for
    for (std::size_t k = 0; k < n; ++k) {
      alignas(64) std::array<ScalarT, block_size> tmp{};
      for (std::size_t i = 0; i < block_size; ++i) {
        ScalarT sum = 0;
#pragma omp simd reduction(+ : sum)
        for (std::size_t j = 0; j < block_size; ++j) sum += data[k * block_size + j] * W.data[i * block_size + j];
        tmp[i] = sum;
      }

#pragma omp simd
      for (std::size_t i = 0; i < block_size; ++i) other.data[k * block_size + i] = tmp[i];
    }
  }

  void dot(BlockView other, MatrixBlockView R)
  {
    R.set_zero();

#pragma omp parallel
    {
      alignas(64) std::array<ScalarT, block_size * block_size> R_temp{};
#pragma omp for
      for (std::size_t k = 0; k < n - 3; k += 4) {
        for (std::size_t i = 0; i < block_size; ++i)
#pragma omp simd
          for (std::size_t j = 0; j < block_size; ++j) {
            R_temp[i * block_size + j] += data[(k + 0) * block_size + i] * other.data[(k + 0) * block_size + j] + data[(k + 1) * block_size + i] * other.data[(k + 1) * block_size + j] +
                                          data[(k + 2) * block_size + i] * other.data[(k + 2) * block_size + j] + data[(k + 3) * block_size + i] * other.data[(k + 3) * block_size + j];
          }
      }

#pragma omp critical
      {
        for (std::size_t i = 0; i < block_size * block_size; ++i) R.data[i] += R_temp[i];
      }
    }

    // Sequential epilogue for remaining rows
    for (std::size_t k = (n / 4) * 4; k < n; ++k) {
      for (std::size_t i = 0; i < block_size; ++i)
#pragma omp simd
        for (std::size_t j = 0; j < block_size; ++j) R.data[i * block_size + j] += data[k * block_size + i] * other.data[k * block_size + j];
    }
  }

  void subtract_product(BlockView other, MatrixBlockView R)
  {
    // this -= other * R
#pragma omp parallel for
    for (std::size_t k = 0; k < n; ++k) {
      for (std::size_t i = 0; i < block_size; ++i) {
        ScalarT sum = 0;
#pragma omp simd reduction(+ : sum)
        for (std::size_t j = 0; j < block_size; ++j) sum += other.data[k * block_size + j] * R.data[j * block_size + i];
        data[k * block_size + i] -= sum;
      }
    }
  }

  ScalarT* data;

private:
  std::size_t n;
};
} // namespace trl::openmp
