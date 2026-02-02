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

    Cv.set_zero();

    // Choose tile size to balance register pressure vs parallelism:
    // - For M*N <= 16: use full M×N (1 tile)
    // - For M*N = 64 (blocksize 8): use 2×2 tiles → 16 tiles, 4 registers per tile
    // - For M*N = 256 (blocksize 16): use 4×4 tiles → 16 tiles, 16 registers per tile
    constexpr unsigned int TM = (M * N <= 16) ? M : ((M * N <= 64) ? 2 : 4);
    constexpr unsigned int TN = (M * N <= 16) ? N : ((M * N <= 64) ? 2 : 4);
    constexpr unsigned int num_tiles_m = (M + TM - 1) / TM;
    constexpr unsigned int num_tiles_n = (N + TN - 1) / TN;
    constexpr unsigned int num_tiles = num_tiles_m * num_tiles_n;

    q->submit([&](::sycl::handler& cgh) {
      auto* A = this->data;
      auto* B = Bv.data;
      auto* C = Cv.data;

      const auto num_cu = q->get_device().get_info<::sycl::info::device::max_compute_units>();
      // More tiles = fewer workers per tile needed to saturate GPU
      const auto workers_per_tile = (num_tiles >= 16) ? 64 * num_cu : 256 * num_cu;
      const auto total_workers = num_tiles * workers_per_tile;

      cgh.parallel_for(::sycl::range<1>(total_workers), [=](auto global_id) {
        auto tile_idx = global_id % num_tiles;
        auto worker_id = global_id / num_tiles;

        auto tile_m = tile_idx / num_tiles_n;
        auto tile_n = tile_idx % num_tiles_n;

        auto m_start = tile_m * TM;
        auto n_start = tile_n * TN;

        T c_local[TM][TN] = {};

        for (std::size_t k = worker_id; k < K; k += workers_per_tile) {
          // Load A row segment into registers
          T a_reg[TM];
          for (unsigned int tm = 0; tm < TM; ++tm) {
            a_reg[tm] = A[k * M + (m_start + tm)];
          }

          // Load B row segment and compute outer product
          for (unsigned int tn = 0; tn < TN; ++tn) {
            T b_val = B[k * N + (n_start + tn)];
            for (unsigned int tm = 0; tm < TM; ++tm) {
              c_local[tm][tn] += a_reg[tm] * b_val;
            }
          }
        }

        // Atomic accumulate to global
        for (unsigned int tm = 0; tm < TM; ++tm) {
          for (unsigned int tn = 0; tn < TN; ++tn) {
            ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space> c_ref(C[(m_start + tm) * N + (n_start + tn)]);
            c_ref += c_local[tm][tn];
          }
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
  std::size_t rows_;
};
} // namespace trl::sycl
