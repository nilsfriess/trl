#pragma once

#include <cassert>
#include <cstddef>

#include <sycl/sycl.hpp>

#include "matrixblockview.hh"

namespace trl::sycl {
// TODO: cols_ should be an unsigned int
template <class T, std::size_t cols_>
class BlockView {
public:
  using EntryType = T;
  using MatrixBlockView = MatrixBlockView<T, cols_>;

  BlockView(::sycl::queue* queue, T* data, std::size_t rows)
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
  std::size_t cols() const { return cols_; }

  /** @brief Copy data from another block view
   *
   *  Copies the data from source into this view. Both views must have
   *  the same dimensions.
   */
  void copy_from(const BlockView& source)
  {
    assert(rows_ == source.rows_);
    q->memcpy(data, source.data, rows_ * cols_ * sizeof(T)).wait();
  }

  void dot(BlockView Bv, MatrixBlockView Cv)
  {
    // C = A^T * B where A is K×M, B is K×N, C is M×N
    const auto K = rows();
    constexpr auto M = cols_;
    constexpr auto N = cols_;

    // For larger blocksizes, split output matrix across work-items within a work-group
    // Each work-item handles a subset of output elements to reduce register pressure
    constexpr unsigned int wg_size = 64;
    constexpr unsigned int elems_per_wi = (M * N + wg_size - 1) / wg_size;
    constexpr unsigned int actual_wis = (M * N + elems_per_wi - 1) / elems_per_wi;

    Cv.set_zero();
    q->submit([&](::sycl::handler& cgh) {
      auto* A = this->data;
      auto* B = Bv.data;
      auto* C = Cv.data;

      const auto num_wg = 64 * q->get_device().get_info<::sycl::info::device::max_compute_units>();

      cgh.parallel_for(
          ::sycl::nd_range<1>(num_wg * wg_size, wg_size), [=](::sycl::nd_item<1> item) {
            auto wg_id = item.get_group_linear_id();
            auto local_id = item.get_local_linear_id();

            // Work-items with local_id >= actual_wis don't contribute output elements
            // but still participate in loading for coalesced access
            auto my_start = local_id * elems_per_wi;
            auto my_end = (my_start + elems_per_wi < M * N) ? my_start + elems_per_wi : M * N;
            if (local_id >= actual_wis) {
              my_start = 0;
              my_end = 0;
            }

            T c_local[elems_per_wi] = {};

            for (std::size_t k = wg_id; k < K; k += num_wg) {
              // Each work-item accumulates its assigned output elements
              for (unsigned int e = 0; e < my_end - my_start; ++e) {
                auto elem = my_start + e;
                auto m = elem / N;
                auto n = elem % N;
                c_local[e] += A[k * M + m] * B[k * N + n];
              }
            }

            // Atomic accumulate to global
            for (unsigned int e = 0; e < my_end - my_start; ++e) {
              auto elem = my_start + e;
              auto m = elem / N;
              auto n = elem % N;
              ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space> c_ref(C[m * N + n]);
              c_ref += c_local[e];
            }
          });
    });
  }

  void mult(MatrixBlockView B, BlockView C)
  {
    const auto n = rows();

    q->submit([&](auto& cgh) {
      auto* a = data;
      auto* b = B.data;
      auto* c = C.data;

      cgh.parallel_for(::sycl::range{n}, [=](::sycl::id<1> id) {
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

      cgh.parallel_for(::sycl::range{n}, [=](::sycl::id<1> id) {
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

      cgh.parallel_for(::sycl::range{n}, [=](::sycl::id<1> id) {
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

      cgh.parallel_for(::sycl::range{n}, [=](::sycl::id<1> id) {
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

    q->submit([&](::sycl::handler& cgh) {
      auto* a = data; // this
      auto* b = B.data;
      auto* c = C.data;

      cgh.parallel_for(::sycl::range<1>{K}, [=](::sycl::id<1> id) {
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

  T norm() const
  {
    T result = 0;
    assert(false && "not implemented");
    return result;
  }

  void set_zero() { q->memset(data, 0, rows_ * cols_ * sizeof(T)); }

  T& operator()(std::size_t row, std::size_t col)
  {
    assert(false && "not implemented");
    return data[row * cols_ + col];
  }

  T* data;

private:
  ::sycl::queue* q;
  const std::size_t rows_;
};
} // namespace trl::sycl
