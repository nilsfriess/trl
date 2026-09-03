#pragma once

#include <cassert>
#include <cstddef>
#include <tuple>
#include <utility>

#include <sycl/sycl.hpp>

#include "../../helpers.hh"
#include "matrixblockview.hh"

inline constexpr std::size_t to_linear_index(std::size_t cols, std::size_t i, std::size_t j) { return i * cols + j; }

namespace trl::Sycl {
// TODO: cols_ should be an unsigned int
/** @brief SYCL block view backed by USM shared memory.
 *
 *  Backend specifics:
 *  - Uses a sycl::queue for all operations; kernels are enqueued on that queue.
 *  - Assumes an in-order queue for correctness of dependent operations.
 *    If an out-of-order queue is used, explicit events are required.
 *  - Methods typically do not wait; synchronization is deferred to the caller.
 */
template <class T, std::size_t cols_>
class BlockView {
public:
  using EntryType = T;
  using MatrixBlockView = MatrixBlockView<T, cols_>;

  BlockView(sycl::queue* queue, T* data, std::size_t rows)
      : data(data)
      , q(queue)
      , rows_(rows)
  {
  }

  // Default copy operations (copying a view is cheap)
  BlockView(const BlockView&) = default;

  BlockView& operator=(const BlockView&) = default;

  // Default move operations
  BlockView(BlockView&&) = default;

  BlockView& operator=(BlockView&&) = default;

  // Default destructor (view doesn't own data)
  ~BlockView() = default;

  std::size_t rows() const { return rows_; }
  constexpr std::size_t cols() const { return cols_; }

  void copy_from(const BlockView& source)
  {
    assert(rows_ == source.rows_);
    q->memcpy(data, source.data, rows_ * cols_ * sizeof(T));
  }

  void dot(BlockView B, MatrixBlockView C)
  {
    const auto n = rows();

    if constexpr (cols_ == 1) {
      auto* a = data;
      auto* b = B.data;
      auto* c = C.data;

      q->submit([&](sycl::handler& h) {
        auto red = sycl::reduction(c, sycl::plus<>(), {sycl::property::reduction::initialize_to_identity{}});
        h.parallel_for(sycl::range<1>(n), red, [a, b](sycl::id<1> i, auto& tmp) {
          double prod = a[i] * b[i];
          tmp += prod;
        });
      });
    }
    else {
      TRL_TODO("TODO");
    }
  }

  void mult(MatrixBlockView B, BlockView C)
  {
    const auto n = rows();

    q->submit([&](auto& cgh) {
      auto* a = data;
      auto* b = B.data;
      auto* c = C.data;

      cgh.parallel_for(sycl::range{n}, [=](sycl::id<1> id) {
        auto tid = id[0];

        T a_private[cols_];
        for (std::size_t i = 0; i < cols_; ++i) a_private[i] = a[tid * cols_ + i];

        T c_private[cols_];
        for (std::size_t i = 0; i < cols_; ++i) {
          T sum{0};
          for (std::size_t j = 0; j < cols_; ++j) sum += a_private[j] * b[j * cols_ + i];
          c_private[i] = sum;
        }

        for (std::size_t i = 0; i < cols_; ++i) c[tid * cols_ + i] = c_private[i];
      });
    });
  }

  void mult_add(MatrixBlockView B, BlockView C)
  {
    const auto n = rows();

    q->submit([&](auto& cgh) {
      auto* a = data;
      auto* b = B.data;
      auto* c = C.data;

      cgh.parallel_for(sycl::range{n}, [=](sycl::id<1> id) {
        auto tid = id[0];

        T a_private[cols_];
        for (std::size_t i = 0; i < cols_; ++i) a_private[i] = a[tid * cols_ + i];

        T c_private[cols_];
        for (std::size_t i = 0; i < cols_; ++i) {
          T sum{0};
          for (std::size_t j = 0; j < cols_; ++j) sum += a_private[j] * b[j * cols_ + i];
          c_private[i] = sum;
        }

        for (std::size_t i = 0; i < cols_; ++i) c[tid * cols_ + i] += c_private[i];
      });
    });
  }

  void mult_transpose(MatrixBlockView B, BlockView C)
  {
    const auto n = rows();

    q->submit([&](auto& cgh) {
      auto* a = data;
      auto* b = B.data;
      auto* c = C.data;

      cgh.parallel_for(sycl::range{n}, [=](sycl::id<1> id) {
        auto tid = id[0];

        T a_private[cols_];
        for (std::size_t i = 0; i < cols_; ++i) a_private[i] = a[tid * cols_ + i];

        T c_private[cols_];
        for (std::size_t i = 0; i < cols_; ++i) {
          T sum{0};
          for (std::size_t j = 0; j < cols_; ++j) sum += a_private[j] * b[i * cols_ + j];
          c_private[i] = sum;
        }

        for (std::size_t i = 0; i < cols_; ++i) c[tid * cols_ + i] = c_private[i];
      });
    });
  }

  BlockView& operator-=(BlockView B)
  {
    const auto n = rows();

    q->submit([&](auto& cgh) {
      auto* a = data;
      auto* b = B.data;

      cgh.parallel_for(sycl::range{n}, [=](sycl::id<1> id) {
        auto tid = id[0];
        for (std::size_t i = 0; i < cols_; ++i) a[tid * cols_ + i] -= b[tid * cols_ + i];
      });
    });

    return *this;
  }

  void subtract_product(BlockView B, MatrixBlockView C)
  {
    const std::size_t K = rows();
    const std::size_t M = cols_;

    q->submit([&](sycl::handler& cgh) {
      auto* a = data; // this
      auto* b = B.data;
      auto* c = C.data;

      cgh.parallel_for(sycl::range<1>{K}, [=](sycl::id<1> id) {
        const std::size_t k = id[0];

        T brow[cols_];
        for (std::size_t m = 0; m < M; ++m) brow[m] = b[k * M + m];

        for (std::size_t n = 0; n < M; ++n) {
          T sum = T(0);
          for (std::size_t m = 0; m < M; ++m) sum += brow[m] * c[m * M + n];

          a[k * M + n] -= sum;
        }
      });
    });
  }

  void set_zero() { q->memset(data, 0, rows_ * cols_ * sizeof(T)); }

  T* data;

private:
  sycl::queue* q;
  const std::size_t rows_;
};
} // namespace trl::Sycl
