#pragma once

#include "util.hh"

#include <algorithm>
#include <span>

namespace trl::openmp {
template <class ScalarT, unsigned int block_size>
class BlockMatrixBlockView {
public:
  using EntryType = ScalarT;

  BlockMatrixBlockView(ScalarT* data)
      : data_(data)
  {
  }

  void copy_from(BlockMatrixBlockView other)
  {
    for (unsigned int i = 0; i < block_size * block_size; ++i) data_[i] = other.data_[i];
  }

  void copy_from_transpose(BlockMatrixBlockView other)
  {
    for (unsigned int i = 0; i < block_size; ++i)
      for (unsigned int j = 0; j < block_size; ++j) data_[i * block_size + j] = other.data_[j * block_size + i];
  }

  void set_zero() { std::fill_n(data_, block_size * block_size, 0); } //

  void mult(BlockMatrixBlockView B, BlockMatrixBlockView C)
  {
    C.set_zero();
    for (unsigned int i = 0; i < block_size; ++i)
      for (unsigned int j = 0; j < block_size; ++j) {
        ScalarT sum = 0;
        for (unsigned int k = 0; k < block_size; ++k) sum += data_[i * block_size + k] * B.data_[k * block_size + j];
        C.data_[i * block_size + j] = sum;
      }
  }

  void set_diagonal(std::span<ScalarT, block_size> values)
  {
    for (unsigned int i = 0; i < block_size; ++i) data_[i * block_size + i] = values[i];
  }

  ScalarT* data_;
};

} // namespace trl::openmp
