#pragma once

#include "util.hh"

#include <algorithm>
#include <span>

namespace trl::openmp {
/** @brief OpenMP matrix block view backed by aligned host memory.
 *
 *  Backend specifics:
 *  - Stores a raw pointer to row-major data.
 *  - All operations are executed sequentially on the CPU.
 *  - No parallelism within block operations (blocks are typically small).
 */
template <class ScalarT, unsigned int block_size>
class BlockMatrixBlockView {
public:
  using EntryType = ScalarT;

  BlockMatrixBlockView(ScalarT* data)
      : data_(data)
  {
  }

  /** @brief Copies matrix data from another block.
   *
   *  Computes the element-wise assignment \f$ A = B \f$.
   */
  void copy_from(BlockMatrixBlockView other)
  {
    for (unsigned int i = 0; i < block_size * block_size; ++i) data_[i] = other.data_[i];
  }

  /** @brief Copies transposed matrix data from another block.
   *
   *  Computes \f$ A = B^T \f$ where A is this block.
   */
  void copy_from_transpose(BlockMatrixBlockView other)
  {
    for (unsigned int i = 0; i < block_size; ++i)
      for (unsigned int j = 0; j < block_size; ++j) data_[i * block_size + j] = other.data_[j * block_size + i];
  }

  /** @brief Sets all matrix entries to zero.
   *
   *  Computes \f$ A_{ij} = 0 \f$ for all \f$ i, j \f$.
   */
  void set_zero() { std::fill_n(data_, block_size * block_size, 0); }

  /** @brief Computes matrix-matrix product.
   *
   *  Computes \f$ C = A B \f$ where A is this block matrix.
   */
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

  /** @brief Sets the diagonal entries of the matrix.
   *
   *  Sets \f$ A_{ii} \f$ to values[i] for all diagonal indices \f$ i \f$.
   *  Off-diagonal entries remain unchanged.
   */
  void set_diagonal(std::span<ScalarT> values)
  {
    for (unsigned int i = 0; i < block_size; ++i) data_[i * block_size + i] = values[i];
  }

  ScalarT* data_;
};

} // namespace trl::openmp
