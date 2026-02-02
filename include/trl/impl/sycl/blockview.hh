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

  template <int worker_mult = 64>
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
      // worker_mult controls parallelism: higher = more workers, less work per thread
      constexpr std::size_t wg_size = 256;
      const auto num_wg_per_tile = std::max<std::size_t>(1, worker_mult * num_cu / wg_size);
      const auto workers_per_tile = num_wg_per_tile * wg_size;
      const auto total_workers = num_tiles * workers_per_tile;

      cgh.parallel_for(
          ::sycl::nd_range<1>(total_workers, wg_size), [=](::sycl::nd_item<1> item) {
            auto global_id = item.get_global_linear_id();
            auto tile_idx = global_id % num_tiles;
            auto worker_id = global_id / num_tiles;

            // Transposed tile mapping: instead of thread ti handling columns [ti*TM, ti*TM+TM),
            // thread ti handles columns ti, ti+num_tiles_m, ti+2*num_tiles_m, ...
            // This makes adjacent threads access adjacent memory locations (coalesced).
            auto tile_m = tile_idx / num_tiles_n;
            auto tile_n = tile_idx % num_tiles_n;

            // With transposed mapping, compute the M indices this tile handles
            // tile_m selects which "group" of interleaved columns
            // Elements: tile_m, tile_m + num_tiles_m, tile_m + 2*num_tiles_m, ...
            unsigned int m_indices[TM];
            unsigned int n_indices[TN];
            if constexpr (TM < M) {
              // Transposed: interleaved pattern for coalesced access
              for (unsigned int tm = 0; tm < TM; ++tm) {
                m_indices[tm] = tile_m + tm * num_tiles_m;
              }
              for (unsigned int tn = 0; tn < TN; ++tn) {
                n_indices[tn] = tile_n + tn * num_tiles_n;
              }
            } else {
              // Small blocksize: contiguous (no transpose needed)
              for (unsigned int tm = 0; tm < TM; ++tm) {
                m_indices[tm] = tm;
              }
              for (unsigned int tn = 0; tn < TN; ++tn) {
                n_indices[tn] = tn;
              }
            }

            T c_local[TM][TN] = {};

            // Use different strategies based on tile size (compile-time known)
            if constexpr (TM >= 4) {
              // Leap-frogging (double buffering) for larger tiles: load next iteration's
              // data while computing on current data to hide memory latency.
              T a_curr[TM], a_next[TM];
              T b_curr[TN], b_next[TN];

              // Prefetch first iteration
              std::size_t k = worker_id;
              if (k < K) {
                for (unsigned int tm = 0; tm < TM; ++tm) {
                  a_next[tm] = A[k * M + m_indices[tm]];
                }
                for (unsigned int tn = 0; tn < TN; ++tn) {
                  b_next[tn] = B[k * N + n_indices[tn]];
                }
              }

              // Main loop with prefetching
              for (; k < K; k += workers_per_tile) {
                // Current = what we prefetched
                for (unsigned int tm = 0; tm < TM; ++tm) a_curr[tm] = a_next[tm];
                for (unsigned int tn = 0; tn < TN; ++tn) b_curr[tn] = b_next[tn];

                // Prefetch next iteration (loads issued early, hide behind compute)
                std::size_t k_next = k + workers_per_tile;
                if (k_next < K) {
                  for (unsigned int tm = 0; tm < TM; ++tm) {
                    a_next[tm] = A[k_next * M + m_indices[tm]];
                  }
                  for (unsigned int tn = 0; tn < TN; ++tn) {
                    b_next[tn] = B[k_next * N + n_indices[tn]];
                  }
                }

                // Compute outer product (happens while loads are in flight)
                for (unsigned int tn = 0; tn < TN; ++tn) {
                  for (unsigned int tm = 0; tm < TM; ++tm) {
                    c_local[tm][tn] += a_curr[tm] * b_curr[tn];
                  }
                }
              }
            } else {
              // Simple path for small tiles (TM < 4)
              for (std::size_t k = worker_id; k < K; k += workers_per_tile) {
                T a_reg[TM];
                for (unsigned int tm = 0; tm < TM; ++tm) {
                  a_reg[tm] = A[k * M + m_indices[tm]];
                }

                for (unsigned int tn = 0; tn < TN; ++tn) {
                  T b_val = B[k * N + n_indices[tn]];
                  for (unsigned int tm = 0; tm < TM; ++tm) {
                    c_local[tm][tn] += a_reg[tm] * b_val;
                  }
                }
              }
            }

            // Sub-group reduction: reduce within sub-group first, then only leader does atomic
            auto sg = item.get_sub_group();
            for (unsigned int tm = 0; tm < TM; ++tm) {
              for (unsigned int tn = 0; tn < TN; ++tn) {
                T sum = ::sycl::reduce_over_group(sg, c_local[tm][tn], ::sycl::plus<T>());
                if (sg.get_local_linear_id() == 0) {
                  ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space> c_ref(
                      C[m_indices[tm] * N + n_indices[tn]]);
                  c_ref += sum;
                }
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
