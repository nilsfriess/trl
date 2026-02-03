#pragma once

#include <cassert>
#include <cstddef>

#include <sycl/sycl.hpp>

#include "matrixblockview.hh"

namespace trl::sycl {
template <class T, unsigned int cols_>
class BlockView {
public:
  using EntryType = T;
  using MatrixBlockView = MatrixBlockView<T, cols_>;

  BlockView(::sycl::queue *queue, T *data, std::size_t rows) : data(data), q(queue), rows_(rows) {}

  // Default copy operations (copying a view is cheap)
  BlockView(const BlockView &) = default;
  BlockView &operator=(const BlockView &) = default;

  // Default move operations
  BlockView(BlockView &&) = default;
  BlockView &operator=(BlockView &&) = default;

  // Default destructor (view doesn't own data)
  ~BlockView() = default;

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  /** @brief Copy data from another block view
   *
   *  Copies the data from source into this view. Both views must have
   *  the same dimensions.
   */
  void copy_from(const BlockView &source)
  {
    assert(rows_ == source.rows_);
    q->memcpy(data, source.data, rows_ * cols_ * sizeof(T)).wait();
  }

  template <int stride = 64>
  void dot(BlockView Bv, MatrixBlockView Cv)
  {
    // C = A^T * B where A is K×M, B is K×N, C is M×N
    const auto K = rows();
    constexpr auto M = cols_;
    constexpr auto N = cols_;

    Cv.set_zero();

    // Choose tile size adaptively, max tile size is always 4x4. This means:
    // - For blocksizes 1, 2, 4: just use 1 tile of size 1x1, 2x2, or 4x4 respectively
    // - For blocksize  8:       use 4 tiles, each of size 4x4
    // - For blocksize 16:       use 16 tiles, each of size 4x4
    constexpr unsigned int max_tilesize = 4;
    constexpr auto TM = std::min(max_tilesize, cols_);
    constexpr auto TN = TM; // only square tiles for now
    constexpr auto num_tiles_m = (M + TM - 1) / TM;
    constexpr auto num_tiles_n = (N + TN - 1) / TN;
    constexpr auto num_tiles = num_tiles_m * num_tiles_n;

    static_assert(stride % num_tiles == 0, "Stride must be divisible by number of tiles");

    q->submit([&](::sycl::handler &cgh) {
      auto *A = this->data;
      auto *B = Bv.data;
      auto *C = Cv.data;

      ::sycl::local_accessor<T, 2> local_C({M, N}, cgh);

      cgh.parallel_for(::sycl::nd_range<1>(stride, 32), [=](::sycl::nd_item<1> item) {
        std::size_t tid = item.get_global_linear_id();
        auto midx = (tid / num_tiles_n) % num_tiles_m;
        auto nidx = tid % num_tiles_n;

        T C_private[TM][TN] = {0.};

        if (item.get_local_id()[0] == 0) {
          for (unsigned int m = 0; m < M; ++m)
            for (unsigned int n = 0; n < N; ++n)
              local_C[m][n] = 0;
        }
        item.barrier(::sycl::access::fence_space::local_space);

        for (std::size_t k = tid / num_tiles; k < K; k += stride / num_tiles)
          for (unsigned int tm = 0; tm < TM; ++tm)
            for (unsigned int tn = 0; tn < TN; ++tn) {
              auto m = midx * TM + tm;
              auto n = nidx * TN + tn;
              C_private[tm][tn] += A[k * cols_ + m] * B[k * cols_ + n];
            }

        // Accumulate to shared memory
        for (unsigned int tm = 0; tm < TM; ++tm)
          for (unsigned int tn = 0; tn < TN; ++tn) {
            auto m = midx * TM + tm;
            auto n = nidx * TN + tn;

            ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device, ::sycl::access::address_space::local_space> C_ref(local_C[m][n]);
            C_ref.fetch_add(C_private[tm][tn]);
          }
        item.barrier(::sycl::access::fence_space::local_space);

        // Accumulate to global C. Since we already reduced in within the subgroup, only the first work item has to do this
        if (item.get_local_id()[0] == 0) {
          for (unsigned int m = 0; m < M; ++m)
            for (unsigned int n = 0; n < N; ++n) {
              ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space> C_ref(C[m * cols_ + n]);
              C_ref.fetch_add(local_C[m][n]);
            }
        }
      });
    });
  }

  void mult(MatrixBlockView B, BlockView C)
  {
    const auto n = rows();

    q->submit([&](auto &cgh) {
      auto *a = data;
      auto *b = B.data;
      auto *c = C.data;

      cgh.parallel_for(::sycl::range{n}, [=](::sycl::id<1> id) {
        auto tid = id[0];

        T a_private[cols_];
        for (std::size_t i = 0; i < cols_; ++i)
          a_private[i] = a[tid * cols_ + i];

        T c_private[cols_];
        for (std::size_t i = 0; i < cols_; ++i) {
          T sum{0};
          for (std::size_t j = 0; j < cols_; ++j)
            sum += a_private[j] * b[j * cols_ + i];
          c_private[i] = sum;
        }

        for (std::size_t i = 0; i < cols_; ++i)
          c[tid * cols_ + i] = c_private[i];
      });
    });
  }

  void mult_add(MatrixBlockView B, BlockView C)
  {
    const auto n = rows();

    q->submit([&](auto &cgh) {
      auto *a = data;
      auto *b = B.data;
      auto *c = C.data;

      cgh.parallel_for(::sycl::range{n}, [=](::sycl::id<1> id) {
        auto tid = id[0];

        T a_private[cols_];
        for (std::size_t i = 0; i < cols_; ++i)
          a_private[i] = a[tid * cols_ + i];

        T c_private[cols_];
        for (std::size_t i = 0; i < cols_; ++i) {
          T sum{0};
          for (std::size_t j = 0; j < cols_; ++j)
            sum += a_private[j] * b[j * cols_ + i];
          c_private[i] = sum;
        }

        for (std::size_t i = 0; i < cols_; ++i)
          c[tid * cols_ + i] += c_private[i];
      });
    });
  }

  void mult_transpose(MatrixBlockView B, BlockView C)
  {
    const auto n = rows();

    q->submit([&](auto &cgh) {
      auto *a = data;
      auto *b = B.data;
      auto *c = C.data;

      cgh.parallel_for(::sycl::range{n}, [=](::sycl::id<1> id) {
        auto tid = id[0];

        T a_private[cols_];
        for (std::size_t i = 0; i < cols_; ++i)
          a_private[i] = a[tid * cols_ + i];

        T c_private[cols_];
        for (std::size_t i = 0; i < cols_; ++i) {
          T sum{0};
          for (std::size_t j = 0; j < cols_; ++j)
            sum += a_private[j] * b[i * cols_ + j];
          c_private[i] = sum;
        }

        for (std::size_t i = 0; i < cols_; ++i)
          c[tid * cols_ + i] = c_private[i];
      });
    });
  }

  BlockView &operator-=(BlockView B)
  {
    const auto n = rows();

    q->submit([&](auto &cgh) {
      auto *a = data;
      auto *b = B.data;

      cgh.parallel_for(::sycl::range{n}, [=](::sycl::id<1> id) {
        auto tid = id[0];
        for (std::size_t i = 0; i < cols_; ++i)
          a[tid * cols_ + i] -= b[tid * cols_ + i];
      });
    });

    return *this;
  }

  void subtract_product(BlockView B, MatrixBlockView C)
  {
    const std::size_t K = rows();
    const std::size_t M = cols_;

    q->submit([&](::sycl::handler &cgh) {
      auto *a = data; // this
      auto *b = B.data;
      auto *c = C.data;

      cgh.parallel_for(::sycl::range<1>{K}, [=](::sycl::id<1> id) {
        const std::size_t k = id[0];

        T brow[cols_];
        for (std::size_t m = 0; m < M; ++m)
          brow[m] = b[k * M + m];

        for (std::size_t n = 0; n < M; ++n) {
          T sum = T(0);
          for (std::size_t m = 0; m < M; ++m)
            sum += brow[m] * c[m * M + n];

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

  T &operator()(std::size_t row, std::size_t col)
  {
    assert(false && "not implemented");
    return data[row * cols_ + col];
  }

  T *data;

private:
  ::sycl::queue *q;
  std::size_t rows_;
};
} // namespace trl::sycl
