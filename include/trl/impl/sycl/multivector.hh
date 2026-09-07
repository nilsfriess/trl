#pragma once

#include <cassert>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include "blockmatrix.hh"
#include "blockview.hh"
#include "dense_matrix.hh"
#include "launch_config.hh"
#include "panel_view.hh"

namespace trl::Sycl {
/** @brief SYCL multivector backed by USM shared memory.
 *
 *  Backend specifics:
 *  - Allocates USM shared memory on construction.
 *  - Stores a sycl::queue by value for submissions.
 *  - Assumes an in-order queue for implicit dependency ordering.
 *  - Copy constructor allocates new storage and zero-initializes it; it does
 *    not copy the underlying data.
 */
template <class T, unsigned int bs>
class BlockMultivector {
public:
  using Scalar = T;
  constexpr static unsigned int blocksize = bs;

  using BlockView = PanelView<T, bs>;
  using PanelView = PanelView<T, bs>;
  using BlockMatrix = BlockMatrix<T, bs>;

  BlockMultivector(sycl::queue queue, std::size_t rows, std::size_t cols)
      : queue(queue)
      , rows(rows)
      , blocks_(cols / blocksize)
      , launch(dot_launch(queue.get_device(), rows))
  {
    if (cols % blocksize != 0) throw std::invalid_argument("Number of columns must be divisible by blocksize");

    data = sycl::malloc_device<T>(rows * cols, queue);
    queue.memset(data, 0, rows * cols * sizeof(T)).wait();
    scratch = sycl::malloc_device<T>(launch.num_groups * bs * bs, queue);
  }

  BlockMultivector(const BlockMultivector& other)
      : queue(other.queue)
      , rows(other.rows)
      , blocks_(other.blocks_)
      , launch(other.launch)
  {
    data = sycl::malloc_device<T>(rows * blocks_ * bs, queue);
    queue.memset(data, 0, rows * blocks_ * bs * sizeof(T)).wait();
    scratch = sycl::malloc_device<T>(launch.num_groups * bs * bs, queue);
  }

  BlockMultivector& operator=(const BlockMultivector& other)
  {
    assert(false && "not implemented");
    if (this != &other) {
      sycl::free(data, queue);
      // sycl::free(scratch, queue);
      queue = other.queue;
      rows = other.rows;
      blocks_ = other.blocks_;
      launch = other.launch;
      data = sycl::malloc_device<T>(rows * blocks_ * bs, queue);
      queue.memset(data, 0, rows * blocks_ * bs * sizeof(T)).wait();
      scratch = sycl::malloc_device<T>(launch.num_groups * bs * bs, queue);
    }
    return *this;
  }

  BlockMultivector(BlockMultivector&& other)
      : queue(std::move(other.queue))
      , rows(other.rows)
      , blocks_(other.blocks_)
      , launch(other.launch)
      , data(other.data)
      , scratch(other.scratch)
  {
    other.data = nullptr;
    other.scratch = nullptr;
  }

  BlockMultivector& operator=(BlockMultivector&& other)
  {
    if (this != &other) {
      if (data) sycl::free(data, queue);
      if (scratch) sycl::free(scratch, queue);
      queue = other.queue;
      rows = other.rows;
      blocks_ = other.blocks_;
      launch = other.launch;
      data = other.data;
      scratch = other.scratch;
      other.data = nullptr;
      other.scratch = nullptr;
    }
    return *this;
  }

  ~BlockMultivector()
  {
    if (data) sycl::free(data, queue);
    if (scratch) sycl::free(scratch, queue);
  }

  PanelView block_view(std::size_t block) { return panel_view(block, 1); }

  // PanelView block_view(std::size_t block) const
  // {
  //   assert(block < blocks_);

  //   return {const_cast<sycl::queue*>(&queue), data + block * rows * bs, rows, 1};
  // }

  std::size_t blocks() const { return blocks_; }

  PanelView panel_view(unsigned int first, unsigned int count)
  {
    assert(first < blocks_);
    assert(first + count < blocks_ + 1);
    assert(count >= 1);
    return {&queue, data + first * rows * bs, rows, count, scratch, launch};
  }

private:
  mutable sycl::queue queue;

  std::size_t rows;
  std::size_t blocks_;

  // Launch geometry for the dot kernel, derived from the device once here and
  // handed to every view: it also fixes the size of the scratch buffer below.
  DotLaunch launch;

  T* data;

  // Scratch for the two-phase dot kernel (bs * bs slots per work-group),
  // shared by all block views. View operations are serialized by the in-order
  // queue, so no two kernels can race on it.
  T* scratch = nullptr;
};
} // namespace trl::Sycl
