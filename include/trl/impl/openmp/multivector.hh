#pragma once

#include "blockmatrix.hh"
#include "blockview.hh"
#include "util.hh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace trl::openmp {
template <class ScalarT, unsigned int block_size>
class BlockMultivector {
public:
  using Scalar = ScalarT;
  using BlockView = BlockView<ScalarT, block_size>;
  using BlockMatrix = BlockMatrix<ScalarT, block_size>;
  using MatrixBlockView = typename BlockMatrix::BlockView;

  static constexpr auto blocksize = block_size;

  BlockMultivector(std::size_t rows, unsigned int cols)
      : rows_(rows)
      , blocks_(cols / blocksize)
  {
    if (cols % blocksize != 0) throw std::invalid_argument("Number of columns must be divisible by blocksize");

    const std::size_t count = rows * cols;
    const std::size_t bytes = count * sizeof(Scalar);
    const std::size_t aligned_bytes = (bytes + 63u) & ~std::size_t(63u);
    data_ = static_cast<Scalar*>(std::aligned_alloc(64, aligned_bytes));
    std::fill_n(data_, rows * cols, 0);
  }

  BlockMultivector(const BlockMultivector&) = delete;
  BlockMultivector& operator=(const BlockMultivector&) = delete;
  BlockMultivector(BlockMultivector&& other)
  {
    rows_ = other.rows_;
    blocks_ = other.blocks_;
    data_ = other.data_;

    other.data_ = nullptr; // Prevent deallocation of moved-from object
  }

  BlockMultivector& operator=(BlockMultivector&& other)
  {
    if (this != &other) {
      std::free(data_);

      rows_ = other.rows_;
      blocks_ = other.blocks_;
      data_ = other.data_;

      other.data_ = nullptr; // Prevent deallocation of moved-from object
    }
    return *this;
  }

  ~BlockMultivector() { std::free(data_); }

  BlockView block_view(std::size_t i) const { return BlockView(data_ + i * rows_ * block_size, rows_); }

  std::size_t blocks() const { return blocks_; }

private:
  std::size_t rows_;
  std::size_t blocks_;
  Scalar* data_;
};
} // namespace trl::openmp
