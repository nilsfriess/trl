#pragma once

#include <cassert>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include <oneapi/math.hpp>

#include "trl/impl/onemath/blockmatrix.hh"
#include "blockview.hh"

namespace trl {
template <class T, unsigned int bs>
class BlockMultivector {
public:
  using Scalar = T;
  constexpr static unsigned int blocksize = bs;

  using BlockView = BlockView<T, bs>;

  using BlockMatrix = BlockMatrix<T, bs>;

  BlockMultivector(sycl::queue queue, std::size_t rows, std::size_t cols)
      : queue(queue)
      , rows(rows)
      , blocks_(cols / blocksize)
  {
    if (cols % blocksize != 0) throw std::invalid_argument("Number of columns must be divisible by blocksize");

    data = sycl::malloc_shared<T>(rows * cols, queue);
    queue.memset(data, 0, rows * cols * sizeof(T)).wait();
  }

  BlockMultivector(const BlockMultivector& other)
      : queue(other.queue)
      , rows(other.rows)
      , blocks_(other.blocks_)
  {
    data = sycl::malloc_shared<T>(rows * blocks_ * bs, queue);
    queue.memset(data, 0, rows * blocks_ * bs * sizeof(T)).wait();
  }

  BlockMultivector& operator=(const BlockMultivector& other)
  {
    if (this != &other) {
      sycl::free(data, queue);
      queue = other.queue;
      rows = other.rows;
      blocks_ = other.blocks_;
      data = sycl::malloc_shared<T>(rows * blocks_ * bs, queue);
      queue.memset(data, 0, rows * blocks_ * bs * sizeof(T)).wait();
    }
    return *this;
  }

  BlockMultivector(BlockMultivector&& other)
      : queue(std::move(other.queue))
      , rows(other.rows)
      , blocks_(other.blocks_)
      , data(other.data)
  {
    other.data = nullptr;
  }

  BlockMultivector& operator=(BlockMultivector&& other)
  {
    if (this != &other) {
      queue = std::move(other.queue);
      rows = other.rows;
      blocks_ = other.blocks_;
      data = other.data;
      other.data = nullptr;
    }
    return *this;
  }

  ~BlockMultivector()
  {
    if (data) sycl::free(data, queue);
  }

  BlockView block_view(std::size_t block)
  {
    assert(block < blocks_);

    return BlockView(&queue, data + block * rows * bs, rows);
  }

  BlockView block_view(std::size_t block) const
  {
    assert(block < blocks_);

    return BlockView(const_cast<sycl::queue*>(&queue), data + block * rows * bs, rows);
  }

  std::size_t blocks() const { return blocks_; }

  /** @brief Subtract another blockvector from this one
   *
   *  Computes this -= other for all blocks.
   */
  BlockMultivector& operator-=(const BlockMultivector& other)
  {
    assert(rows == other.rows);
    assert(blocks_ == other.blocks_);

    for (std::size_t i = 0; i < blocks_; ++i) {
      auto this_block = block_view(i);
      auto other_block = const_cast<BlockMultivector&>(other).block_view(i);
      this_block -= other_block;
    }
    return *this;
  }

  /** @brief Multiply this blockvector by a blockmatrix and store in result
   *
   *  Computes result = this * matrix using all blocks.
   *  Equivalent to calling mult(matrix, result, blocks() - 1).
   *
   *  @param matrix The block matrix to multiply with
   *  @param result The blockvector to store the result
   */
  void mult(const BlockMatrix& matrix, BlockMultivector& result) { mult(matrix, result, blocks_ - 1); }

  /** @brief Multiply this blockvector by a blockmatrix and store in result
   *
   *  Computes result = this * matrix, where:
   *  - this is a blockvector with blocks() blocks (each block is rows × bs)
   *  - matrix is a blockmatrix with block_rows and block_cols
   *  - result is a blockvector
   *
   *  Only blocks 0 to to_block (inclusive) are involved in the computation.
   *  Each block of the result is computed as:
   *  result_block[j] = sum_{i=0}^{to_block} (this_block[i] * matrix_block[i,j])
   *
   *  Performance optimized: queues all GEMM operations before synchronization.
   *
   *  @param matrix The block matrix to multiply with
   *  @param result The blockvector to store the result
   *  @param to_block Last block index to include (inclusive), must be < blocks()
   */
  void mult(const BlockMatrix& matrix, BlockMultivector& result, std::size_t to_block)
  {
    // Verify dimension constraints
    assert(to_block < blocks_);
    assert(to_block < matrix.block_rows());
    assert(matrix.block_cols() <= result.blocks());
    assert(rows == result.rows);

    if (rows != result.rows) throw std::invalid_argument("Row dimensions must match");

    // Queue all gemm operations without waiting (for kernel fusion)
    std::vector<sycl::event> events;
    events.reserve((to_block + 1) * matrix.block_cols());

    // For each output block column
    for (std::size_t j = 0; j < matrix.block_cols(); ++j) {
      T* result_ptr = result.data + j * rows * bs;

      // For each input block row (0 to to_block inclusive)
      for (std::size_t i = 0; i <= to_block; ++i) {
        T* input_ptr = data + i * rows * bs;
        T* matrix_ptr = matrix.data_ + (i * matrix.block_cols() + j) * bs * bs;

        // Queue GEMM: result[j] += this[i] * matrix[i,j]
        // Use beta=0 for first iteration (i=0), beta=1 for accumulation (i>0)
        events.push_back(oneapi::math::blas::row_major::gemm(queue, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, rows, bs, bs, T(1.0), input_ptr, bs, matrix_ptr, bs,
                                                             (i == 0) ? T(0.0) : T(1.0), // beta: 0 for first, 1 for accumulation
                                                             result_ptr, bs));
      }
    }

    // Single synchronization point after all operations are queued
    sycl::event::wait(events);
  }

private:
  sycl::queue queue;

  std::size_t rows;
  std::size_t blocks_;

  T* data;
};
} // namespace trl
