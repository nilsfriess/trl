#pragma once

#include "profiling.hh"

#include <cassert>
#include <span>

#include <sycl/sycl.hpp>

namespace trl::sycl {
/** @brief SYCL matrix block view backed by USM shared memory.
 *
 *  All operations are submitted as device kernels; no host access occurs
 *  inside the methods. Assumes an in-order queue for implicit dependency ordering.
 */
template <class T, unsigned int bs>
class MatrixBlockView {
public:
  using EntryType = T;
  static constexpr unsigned int rows = bs;
  static constexpr unsigned int cols = bs;

  MatrixBlockView(::sycl::queue* queue, T* data, T* diag_scratch)
      : data(data)
      , queue(queue)
      , diag_scratch(diag_scratch)
  {
  }

  MatrixBlockView(const MatrixBlockView&) = default;
  MatrixBlockView& operator=(const MatrixBlockView&) = default;
  MatrixBlockView(MatrixBlockView&&) = default;
  MatrixBlockView& operator=(MatrixBlockView&&) = default;
  ~MatrixBlockView() = default;

  void copy_from(const MatrixBlockView& source)
  {
    static auto* ev = SyclProfiler::get().registerOrGetEvent(
        SyclProfiler::get().registerOrGetFamily("MatrixBlockView"), "copy_from");
    auto e = queue->memcpy(data, source.data, bs * bs * sizeof(T));
    SyclProfiler::get().pushEvent(ev, e);
  }

  void copy_from_transpose(const MatrixBlockView& source)
  {
    static auto* ev = SyclProfiler::get().registerOrGetEvent(
        SyclProfiler::get().registerOrGetFamily("MatrixBlockView"), "copy_from_transpose");
    T* dest_ptr = data;
    T* src_ptr = source.data;
    auto e = queue->single_task([=]() {
      for (std::size_t i = 0; i < bs; ++i)
        for (std::size_t j = 0; j < bs; ++j) dest_ptr[i * bs + j] = src_ptr[j * bs + i];
    });
    SyclProfiler::get().pushEvent(ev, e);
  }

  void set_zero()
  {
    static auto* ev = SyclProfiler::get().registerOrGetEvent(
        SyclProfiler::get().registerOrGetFamily("MatrixBlockView"), "set_zero");
    auto e = queue->memset(data, 0, bs * bs * sizeof(T));
    SyclProfiler::get().pushEvent(ev, e);
  }

  void set_diagonal(std::span<T> values)
  {
    static auto* ev = SyclProfiler::get().registerOrGetEvent(
        SyclProfiler::get().registerOrGetFamily("MatrixBlockView"), "set_diagonal");
    queue->memcpy(diag_scratch, values.data(), bs * sizeof(T));
    T* dest_ptr = data;
    T* scratch = diag_scratch;
    auto e = queue->single_task([=]() {
      for (std::size_t i = 0; i < bs; ++i) dest_ptr[i * bs + i] = scratch[i];
    });
    SyclProfiler::get().pushEvent(ev, e);
  }

  void mult(MatrixBlockView B, MatrixBlockView C)
  {
    static auto* ev = SyclProfiler::get().registerOrGetEvent(
        SyclProfiler::get().registerOrGetFamily("MatrixBlockView"), "mult");
    T* a = data;
    T* b = B.data;
    T* c = C.data;
    auto e = queue->single_task([=]() {
      for (std::size_t i = 0; i < bs; ++i)
        for (std::size_t j = 0; j < bs; ++j) {
          T sum = 0;
          for (std::size_t k = 0; k < bs; ++k) sum += a[i * bs + k] * b[k * bs + j];
          c[i * bs + j] = sum;
        }
    });
    SyclProfiler::get().pushEvent(ev, e);
  }

  // Host-side element access; call queue->wait() before using after device ops.
  T& operator()(std::size_t row, std::size_t col)
  {
    assert(row < bs);
    assert(col < bs);
    return data[row * bs + col];
  }

  T* data;

private:
  ::sycl::queue* queue;
  T* diag_scratch;
};

} // namespace trl::sycl
