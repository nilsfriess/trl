#pragma once

#include "panel_view.hh"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace trl::openmp {
/** @brief OpenMP multivector backed by aligned host memory.
 *
 *  Backend specifics:
 *  - Allocates 64-byte aligned memory using std::aligned_alloc.
 *  - Blocks are stored contiguously in row-major order.
 *  - Zero-initialized on construction.
 *  - Not copyable, only movable.
 */
template <class T, unsigned int bs>
class BlockMultivector {
public:
  using Scalar = T;
  static constexpr unsigned int blocksize = bs;

  using BlockView = PanelView<T, bs>;
  using PanelView = PanelView<T, bs>;

  /** @brief Allocates a block multivector of the given size.
   *
   *  The allocation size is rounded up to the next multiple of the 64-byte
   *  alignment, as std::aligned_alloc requires.
   *
   *  @throws std::invalid_argument if @p cols is not divisible by blocksize.
   */
  BlockMultivector(std::size_t rows, unsigned int cols)
      : rows_(rows)
      , blocks_(cols / blocksize)
  {
    if (cols % blocksize != 0) throw std::invalid_argument("Number of columns must be divisible by blocksize");

    const std::size_t count = rows * cols;
    const std::size_t aligned_bytes = (count * sizeof(T) + 63u) & ~std::size_t(63u);
    data_ = static_cast<T*>(std::aligned_alloc(64, aligned_bytes));
    for (std::size_t i = 0; i < count; ++i) data_[i] = T{0};
  }

  BlockMultivector(const BlockMultivector&) = delete;
  BlockMultivector& operator=(const BlockMultivector&) = delete;

  BlockMultivector(BlockMultivector&& other) noexcept
      : rows_(other.rows_)
      , blocks_(other.blocks_)
      , data_(other.data_)
  {
    other.data_ = nullptr;
  }

  BlockMultivector& operator=(BlockMultivector&& other) noexcept
  {
    if (this != &other) {
      std::free(data_);

      rows_ = other.rows_;
      blocks_ = other.blocks_;
      data_ = other.data_;

      other.data_ = nullptr;
    }
    return *this;
  }

  ~BlockMultivector() { std::free(data_); }

  PanelView block_view(std::size_t block) { return panel_view(static_cast<unsigned int>(block), 1); }

  PanelView panel_view(unsigned int first, unsigned int count)
  {
    assert(first < blocks_);
    assert(first + count < blocks_ + 1);
    assert(count >= 1);
    return {data_ + std::size_t(first) * rows_ * bs, rows_, count};
  }

  std::size_t blocks() const { return blocks_; }

private:
  std::size_t rows_;
  std::size_t blocks_;
  T* data_;
};
} // namespace trl::openmp
