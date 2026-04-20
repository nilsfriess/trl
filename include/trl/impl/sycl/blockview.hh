#pragma once

#include "matrixblockview.hh"

#include <cassert>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <tuple>
#include <utility>

inline constexpr std::size_t to_linear_index(std::size_t cols, std::size_t i, std::size_t j) { return i * cols + j; }

namespace trl::sycl {
// TODO: cols_ should be an unsigned int
/** @brief SYCL block view backed by USM shared memory.
 *
 *  Backend specifics:
 *  - Uses a ::sycl::queue for all operations; kernels are enqueued on that queue.
 *  - Assumes an in-order queue for correctness of dependent operations.
 *    If an out-of-order queue is used, explicit events are required.
 *  - Methods typically do not wait; synchronization is deferred to the caller.
 */
template <class T, std::size_t cols_>
class BlockView {
public:
  using EntryType = T;
  using MatrixBlockView = MatrixBlockView<T, cols_>;

  BlockView(::sycl::queue* queue, T* data, std::size_t rows)
      : data(data)
      , q(queue)
      , rows_(rows)
      , global_size_(128 * cols_ * queue->get_device().get_info<::sycl::info::device::max_compute_units>())
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

  void copy_from(const BlockView& source)
  {
    assert(rows_ == source.rows_);
    q->memcpy(data, source.data, rows_ * cols_ * sizeof(T));
  }

  template <bool transposed = true>
  void dot(BlockView B, MatrixBlockView C)
  {
    C.set_zero();

    constexpr auto TNTM = []() {
      if constexpr (cols_ == 1) return std::make_pair<unsigned int, unsigned int>(1, 1);
      else if constexpr (cols_ == 2) return std::make_pair<unsigned int, unsigned int>(2, 2);
      else if constexpr (cols_ == 4) return std::make_pair<unsigned int, unsigned int>(2, 2);
      else if constexpr (cols_ == 8) return std::make_pair<unsigned int, unsigned int>(4, 4);
      else if constexpr (cols_ >= 16) return std::make_pair<unsigned int, unsigned int>(4, 4);
      else return std::make_pair<unsigned int, unsigned int>(cols_, 1);
    }();
    constexpr auto TN = std::get<0>(TNTM);
    constexpr auto TM = std::get<1>(TNTM);

    static_assert(cols_ % TN == 0);
    static_assert(cols_ % TM == 0);
    constexpr auto N = cols_ / TN;
    constexpr auto M = cols_ / TM;

    const auto global_size = global_size_;

    constexpr auto m_index = [=](std::size_t midx, unsigned int tm) {
      if constexpr (transposed) return midx + tm * M;
      else return midx * TM + tm;
    };

    constexpr auto n_index = [=](std::size_t nidx, unsigned int tn) {
      if constexpr (transposed) return nidx + tn * N;
      else return nidx * TN + tn;
    };

    q->submit([&](auto& cgh) {
      const auto* a = data;
      const auto* b = B.data;
      auto* c = C.data;
      const auto rows = rows_;

      cgh.parallel_for(::sycl::range<1>{global_size}, [=](::sycl::item<1> it) {
        const auto tid = it[0];

        const auto tile_id = tid % (N * M);
        const auto worker_id = tid / (N * M);
        const auto num_workers = global_size / (N * M);

        const auto midx = tile_id / N;
        const auto nidx = tile_id % N;

        T C_local[TM][TN] = {};

        for (std::size_t k = worker_id; k < rows; k += num_workers) {
          T a_frag[TM];
          T b_frag[TN];

          for (unsigned int tm = 0; tm < TM; ++tm) {
            const auto m = m_index(midx, tm);
            a_frag[tm] = a[to_linear_index(cols_, k, m)];
          }
          for (unsigned int tn = 0; tn < TN; ++tn) {
            const auto n = n_index(nidx, tn);
            b_frag[tn] = b[to_linear_index(cols_, k, n)];
          }

          for (unsigned int tm = 0; tm < TM; ++tm)
            for (unsigned int tn = 0; tn < TN; ++tn) C_local[tm][tn] += a_frag[tm] * b_frag[tn];
        }

        for (unsigned int tm = 0; tm < TM; ++tm)
          for (unsigned int tn = 0; tn < TN; ++tn) {
            const auto m = m_index(midx, tm);
            const auto n = n_index(nidx, tn);

            ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device> atomic_c(c[to_linear_index(cols_, m, n)]);
            atomic_c += C_local[tm][tn];
          }
      });
    });
  }

  // C = this * B
  void mult(MatrixBlockView B, BlockView C)
  {
    const auto n = rows();
    const auto global_size = global_size_;

    q->submit([&](auto& cgh) {
      auto* a = data;
      auto* b = B.data;
      auto* c = C.data;

      cgh.parallel_for(::sycl::range{global_size}, [=](::sycl::id<1> id) {
        for (std::size_t tid = id[0]; tid < n; tid += global_size) {
          T a_private[cols_];
          for (std::size_t i = 0; i < cols_; ++i) a_private[i] = a[tid * cols_ + i];

          T c_private[cols_];
          for (std::size_t i = 0; i < cols_; ++i) {
            T sum{0};
            for (std::size_t j = 0; j < cols_; ++j) sum += a_private[j] * b[j * cols_ + i];
            c_private[i] = sum;
          }

          for (std::size_t i = 0; i < cols_; ++i) c[tid * cols_ + i] = c_private[i];
        }
      });
    });
  }

  // Implements
  //   this <- alpha * A * B + beta * this
  // or
  //   this <- alpha * A * B^T + beta * this,
  // where A is a blockview of the same size as `this`, and B is a sqaure block matrix view.
  //
  // Since alpha and beta are usually either 0, 1, or -1, we use AdaptiveCpp's sycl::specialized extension
  // so that the JIT compiler can optimise unnecessary computations away.
  // TODO: At ACPP_ADAPTIVITY_LEVEL=2, the compiler will do this automatically, so we might consider just relying on that
  template <bool transposed>
  void gemm(T alpha, BlockView A, MatrixBlockView B, T beta)
  {
    ::sycl::specialized<T> salpha = alpha;
    ::sycl::specialized<T> sbeta = beta;

    const auto n = rows();
    const auto stride = global_size_;

    q->submit([&](auto& cgh) {
      const auto* a = A.data;
      const auto* b = B.data;
      auto* c = data;

      cgh.parallel_for(::sycl::range{stride}, [=](::sycl::id<1> id) {
        for (auto tid = id[0]; tid < n; tid += stride) {
          // Load the current row into registers
          T a_private[cols_];
          for (std::size_t i = 0; i < cols_; ++i) a_private[i] = a[tid * cols_ + i];

          // Compute c_private[:] = a_private[:] * B
          T c_private[cols_];
          for (std::size_t i = 0; i < cols_; ++i) {
            T sum{0};
            for (std::size_t j = 0; j < cols_; ++j)
              if constexpr (transposed) sum += a_private[j] * b[i * cols_ + j];
              else sum += a_private[j] * b[j * cols_ + i];
            c_private[i] = sum;
          }

          // Update c: c[tid, :] = alpha * c_private[:] + beta * c[tid, :]
          for (std::size_t i = 0; i < cols_; ++i) c[tid * cols_ + i] = beta * c[tid * cols_ + i] + alpha * c_private[i];
        }
      });
    });
  }

  void mult_add(MatrixBlockView B, BlockView C)
  {
    const auto n = rows();
    const auto global_size = global_size_;

    q->submit([&](auto& cgh) {
      auto* a = data;
      auto* b = B.data;
      auto* c = C.data;

      cgh.parallel_for(::sycl::range{global_size}, [=](::sycl::id<1> id) {
        for (std::size_t tid = id[0]; tid < n; tid += global_size) {
          T a_private[cols_];
          for (std::size_t i = 0; i < cols_; ++i) a_private[i] = a[tid * cols_ + i];

          T c_private[cols_];
          for (std::size_t i = 0; i < cols_; ++i) {
            T sum{0};
            for (std::size_t j = 0; j < cols_; ++j) sum += a_private[j] * b[j * cols_ + i];
            c_private[i] = sum;
          }

          for (std::size_t i = 0; i < cols_; ++i) c[tid * cols_ + i] += c_private[i];
        }
      });
    });
  }

  void mult_transpose(MatrixBlockView B, BlockView C)
  {
    const auto n = rows();
    const auto global_size = global_size_;

    q->submit([&](auto& cgh) {
      auto* a = data;
      auto* b = B.data;
      auto* c = C.data;

      cgh.parallel_for(::sycl::range{global_size}, [=](::sycl::id<1> id) {
        for (std::size_t tid = id[0]; tid < n; tid += global_size) {
          T a_private[cols_];
          for (std::size_t i = 0; i < cols_; ++i) a_private[i] = a[tid * cols_ + i];

          T c_private[cols_];
          for (std::size_t i = 0; i < cols_; ++i) {
            T sum{0};
            for (std::size_t j = 0; j < cols_; ++j) sum += a_private[j] * b[i * cols_ + j];
            c_private[i] = sum;
          }

          for (std::size_t i = 0; i < cols_; ++i) c[tid * cols_ + i] = c_private[i];
        }
      });
    });
  }

  BlockView& operator-=(BlockView B)
  {
    const auto n = rows();
    const auto global_size = global_size_;

    q->submit([&](auto& cgh) {
      auto* a = data;
      auto* b = B.data;

      cgh.parallel_for(::sycl::range{global_size}, [=](::sycl::id<1> id) {
        for (std::size_t tid = id[0]; tid < n; tid += global_size)
          for (std::size_t i = 0; i < cols_; ++i) a[tid * cols_ + i] -= b[tid * cols_ + i];
      });
    });

    return *this;
  }

  void subtract_product(BlockView B, MatrixBlockView C)
  {
    const std::size_t K = rows();
    const std::size_t M = cols_;
    const auto global_size = global_size_;

    q->submit([&](::sycl::handler& cgh) {
      auto* a = data; // this
      auto* b = B.data;
      auto* c = C.data;

      cgh.parallel_for(::sycl::range<1>{global_size}, [=](::sycl::id<1> id) {
        for (std::size_t k = id[0]; k < K; k += global_size) {
          T brow[cols_];
          for (std::size_t m = 0; m < M; ++m) brow[m] = b[k * M + m];

          for (std::size_t n = 0; n < M; ++n) {
            T sum = T(0);
            for (std::size_t m = 0; m < M; ++m) sum += brow[m] * c[m * M + n];

            a[k * M + n] -= sum;
          }
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
  template <std::size_t... Is>
  static auto make_reductions(T* c, std::index_sequence<Is...>)
  {
    return std::make_tuple(
        ::sycl::reduction(c + Is, T{0}, ::sycl::plus<T>{}, ::sycl::property_list{::sycl::property::reduction::initialize_to_identity{}})...);
  }

  template <std::size_t... Is, class... Reducers>
  static void combine_reductions(const T* a, const T* b, std::size_t m, std::size_t k, std::index_sequence<Is...>, Reducers&... reducers)
  {
    ((reducers.combine(a[to_linear_index(m, k, Is / cols_)] * b[to_linear_index(m, k, Is % cols_)])), ...);
  }

  ::sycl::queue* q;
  const std::size_t rows_;
  const std::size_t global_size_;
};
} // namespace trl::sycl
