#pragma once

#include <cassert>
#include <cstddef>

#include <sycl/sycl.hpp>

#include "matrixblockview.hh"

namespace trl {
template <class T, unsigned int bs>
class BlockMatrix {
public:
  using BlockView = MatrixBlockView<T, bs>;

  BlockMatrix(sycl::queue queue, std::size_t block_rows, std::size_t block_cols)
      : queue(queue)
      , block_rows_(block_rows)
      , block_cols_(block_cols)
  {
    data_ = sycl::malloc_shared<T>(block_rows * block_cols * bs * bs, queue);
    // Zero-initialize the matrix
    queue.memset(data_, 0, block_rows * block_cols * bs * bs * sizeof(T)).wait();
  }

  ~BlockMatrix() { sycl::free(data_, queue); }

  std::size_t block_rows() const { return block_rows_; }
  std::size_t block_cols() const { return block_cols_; }

  BlockView block_view(std::size_t block_row, std::size_t block_col) const
  {
    assert(block_row < block_rows_);
    assert(block_col < block_cols_);
    const auto block_index = block_row * block_cols_ + block_col;
    return BlockView(const_cast<sycl::queue*>(&queue), data_ + block_index * bs * bs);
  }

  // Allow BlockMultivector to access data for optimized multiplication
  template <class, unsigned int>
  friend class BlockMultivector;

  // Public accessor for raw data pointer (needed for LAPACK operations)
  T* data() { return data_; }
  const T* data() const { return data_; }

  void print(bool with_separator = true, bool scientific = true, int precision = 8, T tolerance = 1e-10) const
  {
    if (with_separator) {
      std::cout << "--------------------------------------------------\n";
      std::cout << "Block matrix of order " << block_rows_ << "x" << block_cols_ << "\n";
    }

    // Set output format for high precision
    std::ios_base::fmtflags original_flags = std::cout.flags();
    std::streamsize original_precision = std::cout.precision();

    if (scientific) std::cout << std::scientific << std::setprecision(precision);
    else std::cout << std::fixed << std::setprecision(precision);

    // Calculate field width based on precision and format
    int field_width = scientific ? precision + 7 : precision + 4; // Adjust for format overhead

    for (std::size_t i = 0; i < block_rows_; ++i) {
      // Print each row of blocks
      for (std::size_t bi = 0; bi < bs; ++bi) {
        for (std::size_t j = 0; j < block_cols_; ++j) {
          auto block = block_view(i, j);

          // Print vertical separator before block (except for first block)
          if (j > 0 && with_separator) std::cout << "|";

          for (std::size_t bj = 0; bj < bs; ++bj) {
            auto entry = block(bi, bj);
            if (std::abs(entry) < tolerance) std::cout << std::setw(field_width) << "*";
            else std::cout << std::setw(field_width) << entry;
            if (bj < bs - 1) std::cout << " ";
          }
        }
        std::cout << "\n";
      }

      // Add separator line between block rows
      if (i < block_rows_ - 1 && with_separator) {
        // Print horizontal separator line
        for (std::size_t j = 0; j < block_cols_; ++j) {
          if (j > 0) std::cout << "+";
          for (std::size_t bj = 0; bj < bs; ++bj) {
            std::cout << std::string(field_width, '-');
            if (bj < bs - 1) std::cout << "-";
          }
        }
        std::cout << "\n";
      }
    }

    if (with_separator) std::cout << "--------------------------------------------------\n";

    // Restore original formatting
    std::cout.flags(original_flags);
    std::cout.precision(original_precision);
  }

private:
  sycl::queue queue;

  std::size_t block_rows_;
  std::size_t block_cols_;

  T* data_;
};

} // namespace trl
