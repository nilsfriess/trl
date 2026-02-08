#pragma once

#include <cassert>
#include <cstddef>
#include <tuple>
#include <utility>

#include <sycl/sycl.hpp>

#include "matrixblockview.hh"

inline constexpr std::size_t to_linear_index(std::size_t cols, std::size_t i, std::size_t j) { return i * cols + j; }

namespace trl::sycl {
// TODO: cols_ should be an unsigned int
template <class T, std::size_t cols_>
class BlockView {
  static constexpr size_t local_size = 128;

public:
  using EntryType = T;
  using MatrixBlockView = MatrixBlockView<T, cols_>;

  BlockView(::sycl::queue* queue, T* data, std::size_t rows)
      : data(data)
      , q(queue)
      , rows_(rows)
  {
    // TODO: We implicitly assume that an in-order queue is used, we should check this and throw an error if it's not.
    //       Or use events you express dependencies.

    // Query once at startup
    size_t num_cu = q->get_device().get_info<::sycl::info::device::max_compute_units>();

    size_t num_groups = num_cu * 6;
    global_size = local_size * num_groups;
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
    q->memcpy(data, source.data, rows_ * cols_ * sizeof(T));
  }

  void dot(BlockView B, MatrixBlockView C)
  {
    const auto n = rows();
    const auto m = cols();

    // Zero the output matrix first
    C.set_zero();

    q->submit([&](auto& cgh) {
      auto* a = data;
      auto* b = B.data;
      auto* c = C.data;

      if constexpr (false) {
        auto reductions = make_reductions(c, std::make_index_sequence<cols_ * cols_>{});
        std::apply(
            [&](auto&... rs) {
              cgh.parallel_for(::sycl::range<1>{n}, rs..., [=](::sycl::id<1> id, auto&... reducers) {
                const std::size_t k = id[0];
                combine_reductions(a, b, m, k, std::make_index_sequence<cols_ * cols_>{}, reducers...);
              });
            },
            reductions);
      }
      else {
        cgh.parallel_for(::sycl::range<1>{n}, [=](::sycl::item<1> it) {
          const auto tid = it[0];

          alignas(64) T C_local[cols_][cols_] = {};
          for (unsigned int i = 0; i < m; ++i)
            for (unsigned int j = 0; j < m; ++j) C_local[i][j] += a[to_linear_index(m, tid, i)] * b[to_linear_index(m, tid, j)];

          for (unsigned int i = 0; i < m; ++i)
            for (unsigned int j = 0; j < m; ++j) {
              ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device> atomic_c(c[to_linear_index(m, i, j)]);
              atomic_c += C_local[i][j];
            }
        });
      }
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
  template <std::size_t... Is>
  static auto make_reductions(T* c, std::index_sequence<Is...>)
  {
    return std::make_tuple(::sycl::reduction(c + Is, T{0}, ::sycl::plus<T>{}, ::sycl::property_list{::sycl::property::reduction::initialize_to_identity{}})...);
  }

  template <std::size_t... Is, class... Reducers>
  static void combine_reductions(const T* a, const T* b, std::size_t m, std::size_t k, std::index_sequence<Is...>, Reducers&... reducers)
  {
    ((reducers.combine(a[to_linear_index(m, k, Is / cols_)] * b[to_linear_index(m, k, Is % cols_)])), ...);
  }

  ::sycl::queue* q;
  const std::size_t rows_;

  std::size_t global_size{}; // for dot product
};
} // namespace trl::sycl
