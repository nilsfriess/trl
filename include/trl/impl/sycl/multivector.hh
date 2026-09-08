#pragma once

#include <cassert>
#include <stdexcept>

#include <sycl/sycl.hpp>

#include "launch_config.hh"
#include "panel_view.hh"

namespace trl::Sycl {
/** @brief SYCL multivector backed by USM device memory.
 *
 *  Backend specifics:
 *  - Allocates USM device memory on construction.
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

  BlockMultivector(sycl::queue queue, std::size_t rows, std::size_t cols)
      : queue(queue)
      , rows(rows)
      , blocks_(cols / blocksize)
      , launch(dot_launch(queue.get_device(), rows))
  {
    if (cols % blocksize != 0) throw std::invalid_argument("Number of columns must be divisible by blocksize");

    data = sycl::malloc_device<T>(rows * cols, queue);
    queue.memset(data, 0, rows * cols * sizeof(T)).wait();
  }

  BlockMultivector(const BlockMultivector& other)
      : queue(other.queue)
      , rows(other.rows)
      , blocks_(other.blocks_)
      , launch(other.launch)
  {
    data = sycl::malloc_device<T>(rows * blocks_ * bs, queue);
    queue.memset(data, 0, rows * blocks_ * bs * sizeof(T)).wait();
  }

  BlockMultivector& operator=(const BlockMultivector& other)
  {
    assert(false && "not implemented");
    if (this != &other) {
      sycl::free(data, queue);
      queue = other.queue;
      rows = other.rows;
      blocks_ = other.blocks_;
      launch = other.launch;
      data = sycl::malloc_device<T>(rows * blocks_ * bs, queue);
      queue.memset(data, 0, rows * blocks_ * bs * sizeof(T)).wait();
    }
    return *this;
  }

  BlockMultivector(BlockMultivector&& other)
      : queue(std::move(other.queue))
      , rows(other.rows)
      , blocks_(other.blocks_)
      , launch(other.launch)
      , data(other.data)
  {
    other.data = nullptr;
  }

  BlockMultivector& operator=(BlockMultivector&& other)
  {
    if (this != &other) {
      if (data) sycl::free(data, queue);
      queue = other.queue;
      rows = other.rows;
      blocks_ = other.blocks_;
      launch = other.launch;
      data = other.data;
      other.data = nullptr;
    }
    return *this;
  }

  ~BlockMultivector()
  {
    if (data) sycl::free(data, queue);
  }

  PanelView block_view(std::size_t block) { return panel_view(block, 1); }

  std::size_t blocks() const { return blocks_; }

  PanelView panel_view(unsigned int first, unsigned int count)
  {
    assert(first < blocks_);
    assert(first + count < blocks_ + 1);
    assert(count >= 1);
    return {&queue, data + first * rows * bs, rows, count, launch};
  }

private:
  mutable sycl::queue queue;

  std::size_t rows;
  std::size_t blocks_;

  // Launch geometry for the dot kernel, derived from the device once here and
  // handed to every view.
  DotLaunch launch;

  T* data;
};
} // namespace trl::Sycl
