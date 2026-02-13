#pragma once

#include "blockmatrix.hh"
#include "blockview.hh"
#include "util.hh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace trl::openmp {
  /** @brief OpenMP multivector backed by aligned host memory.
   *
   *  Backend specifics:
   *  - Allocates 64-byte aligned memory using std::aligned_alloc.
   *  - Blocks are stored contiguously in row-major order.
   *  - Zero-initialized on construction.
   *  - Not copyable, only movable.
   */
template <class ScalarT, unsigned int block_size>
class BlockMultivector {
public:
  using Scalar = ScalarT;
  using BlockView = ::trl::openmp::BlockView<ScalarT, block_size>;
  using BlockMatrix = ::trl::openmp::BlockMatrix<ScalarT, block_size>;
  using MatrixBlockView = typename BlockMatrix::BlockView;

  static constexpr auto blocksize = block_size;

  /** @brief Allocates a block multivector of the given size.
   *
   *  Allocates memory aligned to 64 bytes. If rows * cols is not a multiple of alignment,
   *  the next larger multiple is used as the allocation size (this is a requirement of
   *  std::aligned_alloc). The aligned memory is zero-initialised.
   *
   *  @throws std::invalid_argument if \p cols is not divisible by blocksize.
   */
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

  /** @brief Returns a view of the i-th block.
   *
   *  The returned view is a lightweight std::span-like object that does not
   *  own the data.
   *
   *  @throws std::out_of_range if i is not a valid block index.
   */
  BlockView block_view(std::size_t i) const
  {
    if (i >= blocks_) throw std::out_of_range("Block " + std::to_string(i) + " is out of bounds (number of blocks is " + std::to_string(blocks_) + ")");
    return BlockView(data_ + i * rows_ * block_size, rows_);
  }

  /** @brief Returns the number of blocks in the multivector. */
  std::size_t blocks() const { return blocks_; }

private:
  std::size_t rows_;
  std::size_t blocks_;
  Scalar* data_;
};
} // namespace trl::openmp
