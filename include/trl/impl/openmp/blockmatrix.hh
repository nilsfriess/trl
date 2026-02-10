#pragma once

#include "matrixblockview.hh"
#include "util.hh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>

namespace trl::openmp {
/** @brief OpenMP block matrix backed by aligned host memory.
 *
 *  Backend specifics:
 *  - Allocates 64-byte aligned memory using std::aligned_alloc.
 *  - Zero-initialized on construction.
 *  - Not copyable or movable.
 */
template <class ScalarT, unsigned int block_size>
class BlockMatrix {
public:
  using BlockView = BlockMatrixBlockView<ScalarT, block_size>;

  BlockMatrix(unsigned int brows, unsigned int bcols)
      : block_rows_(brows)
      , block_cols_(bcols)
  {
    const std::size_t count = static_cast<std::size_t>(brows) * bcols * block_size * block_size;
    const std::size_t bytes = count * sizeof(ScalarT);
    const std::size_t aligned_bytes = (bytes + 63u) & ~std::size_t(63u);
    data = static_cast<ScalarT*>(std::aligned_alloc(64, aligned_bytes));
    std::fill_n(data, count, ScalarT(0));
  }

  BlockMatrix(const BlockMatrix&) = delete;
  BlockMatrix& operator=(const BlockMatrix&) = delete;
  BlockMatrix(BlockMatrix&&) = delete;
  BlockMatrix& operator=(BlockMatrix&&) = delete;

  ~BlockMatrix() { std::free(data); }

  /** @brief Returns the number of block rows. */
  std::size_t block_rows() const { return block_rows_; }

  /** @brief Returns the number of block columns. */
  std::size_t block_cols() const { return block_cols_; }

  /** @brief Returns a view of the block at position (block_row, block_col).
   *
   *  The returned view is a lightweight object that does not own the data.
   */
  BlockView block_view(std::size_t block_row, std::size_t block_col) const
  {
    assert(block_row < block_rows_);
    assert(block_col < block_cols_);
    const auto block_index = block_row * block_cols_ + block_col;
    return BlockView(data + block_index * block_size * block_size);
  }

private:
  unsigned int block_rows_;
  unsigned int block_cols_;

  ScalarT* data;
};

} // namespace trl::openmp
