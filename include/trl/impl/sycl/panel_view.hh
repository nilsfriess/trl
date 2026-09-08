#pragma once

#include "trl/concepts.hh"
#include "trl/helpers.hh"
#include "trl/impl/sycl/dense_matrix.hh"
#include "trl/impl/sycl/launch_config.hh"

#include <cstddef>
#include <hipSYCL/sycl/libkernel/memory.hpp>
#include <sycl/sycl.hpp>
#include <vector>

namespace trl::Sycl {
template <class T, unsigned int bs>
class PanelView {
public:
  /** @param launch Launch geometry for dot; @p scratch must hold at least
   *  launch.num_groups * bs * bs entries. Both come from the owning
   *  BlockMultivector, which derives them from the device. */
  PanelView(sycl::queue* q_, T* start, std::size_t rows, unsigned int count, T* scratch, DotLaunch launch)
      : q(q_)
      , start_(start)
      , rows_(rows)
      , count_(count)
      , scratch_(scratch)
      , launch_(launch)
  {
  }

  // void set_zero() { q->memset(start_, 0, rows_ * count_ * bs * sizeof(T)); } //

  void copy_from(PanelView other) { TRL_TODO("PanelView::copy_from"); }

  /** @brief Computes out = this * M (or this * M^T for TransposeMode::Transpose).
   *
   *  @p M is (cols() x out.cols()) row-major for NoTranspose and
   *  (out.cols() x cols()) row-major for TransposeMode::Transpose, i.e. the
   *  transpose factor itself is stored rather than computed. If @p events is
   *  given, the events of all submitted kernels are appended to it, as in dot.
   */
  void mult(TransposeMode t, const DenseMatrix<T>& M, PanelView out, std::vector<sycl::event>* events = nullptr)
  {
    TRL_CHECK(rows() == out.rows(), "Input panel and output panel must have the same number of rows");
    if (t == TransposeMode::NoTranspose) {
      TRL_CHECK(cols() == M.rows(), "Number of columns of input panel must match number of rows of matrix");
      TRL_CHECK(M.cols() == out.cols(), "Number of columns of output panel must match number of cols of matrix");
    }
    else {
      TRL_CHECK(out.cols() == M.rows(), "For TransposeMode::Transpose the matrix must be stored as (out.cols() x cols())");
      TRL_CHECK(M.cols() == cols(), "For TransposeMode::Transpose the number of columns of the matrix must match the number of columns of the input panel");
    }

    // When in and out alias, the kernel below only works for single-block panels
    if (out.data() == data()) TRL_CHECK(count_ == 1 && out.count_ == 1, "In-place mult is only supported for single-block panels");

    const T* a = data();
    const T* b = M.data();
    T* c = out.data();
    const auto in_blocks = count_;
    const auto out_blocks = out.count_;
    const auto n = rows();
    const auto ldb = M.cols(); // M is row-major, so its row stride is its column count

    sycl::specialized<bool> tp = (t == TransposeMode::Transpose);

    sycl::event mult_event = q->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        auto tid = id[0];

        for (unsigned int Bo = 0; Bo < out_blocks; ++Bo) {
          T c_private[bs];
          for (unsigned int i = 0; i < bs; ++i) c_private[i] = T{0};

          for (unsigned int Bi = 0; Bi < in_blocks; ++Bi) {
            const T* a_base = a + Bi * n * bs;

            T a_private[bs];
            for (unsigned int j = 0; j < bs; ++j) a_private[j] = a_base[tid * bs + j];

            for (unsigned int i = 0; i < bs; ++i)
              for (unsigned int j = 0; j < bs; ++j)
                if (tp) c_private[i] += a_private[j] * b[(Bo * bs + i) * ldb + (Bi * bs + j)];
                else c_private[i] += a_private[j] * b[(Bi * bs + j) * ldb + (Bo * bs + i)];
          }

          T* c_base = c + Bo * n * bs;
          for (unsigned int i = 0; i < bs; ++i) c_base[tid * bs + i] = c_private[i];
        }
      });
    });
    if (events) events->push_back(mult_event);
  }

  void subtract(PanelView other)
  {
    const auto n = rows();

    q->submit([&](auto& cgh) {
      auto* a = data();
      const auto* b = other.data();
      sycl::specialized<unsigned int> n_blocks(count_);

      cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        auto tid = id[0];
        for (unsigned int B = 0; B < n_blocks; ++B) {
          auto* a_start = a + B * n * bs;
          const auto* b_start = b + B * n * bs;
          for (unsigned int i = 0; i < bs; ++i) a_start[tid * bs + i] -= b_start[tid * bs + i];
        }
      });
    });
  }

  /** @brief Computes out = this^T * Y.
   */
  void dot(PanelView Y, DenseMatrix<T>& out, std::vector<sycl::event>* events = nullptr)
  {
    TRL_CHECK(count_ == 1, "Only the single block panel case is implemented");

    const auto local_size = launch_.local_size;
    const auto global_size = launch_.global_size();
    const bool interleaved = sycl::specialized(launch_.interleaved);

    const T* a = data();
    const T* b = Y.data();
    T* c = out.data();
    // T* s = scratch_;
    const auto n = rows();

    sycl::event memset_event = q->memset(c, 0, sizeof(T) * bs * bs);
    if (events) events->push_back(memset_event);

    sycl::event reduce_event = q->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> it) {
        const std::size_t gid = it.get_global_linear_id();
        const std::size_t gsize = it.get_global_range().size();

        // The two loops differ only in how rows are distributed; the body is
        // repeated rather than hoisted into a lambda or a runtime stride so
        // that the chunked variant keeps its literal stride of 1 (which is
        // what lets the CPU backends vectorize it).
        T sum[bs * bs] = {0};
        if (interleaved) {
          // GPU: consecutive work-items take consecutive rows, so the loads of
          // a sub-group cover one contiguous bs * sizeof(T) * sub_group_size
          // span and coalesce into full transactions.
          for (std::size_t i = gid; i < n; i += gsize)
            for (unsigned int I = 0; I < bs; ++I)
              for (unsigned int J = 0; J < bs; ++J) sum[I * bs + J] += a[i * bs + I] * b[i * bs + J];
        }
        else {
          // CPU: one contiguous, prefetchable chunk of rows per work-item.
          const std::size_t chunk = (n + gsize - 1) / gsize;
          const std::size_t begin = sycl::min(gid * chunk, n);
          const std::size_t end = sycl::min(begin + chunk, n);
          for (std::size_t i = begin; i < end; ++i)
            for (unsigned int I = 0; I < bs; ++I)
              for (unsigned int J = 0; J < bs; ++J) sum[I * bs + J] += a[i * bs + I] * b[i * bs + J];
        }

        T reduced_sums[bs * bs];
        // Note: Do not nest this loop. A compiler bug in AdaptiveCpp miscompiles this (see https://github.com/AdaptiveCpp/AdaptiveCpp/issues/2224)
        for (unsigned int K = 0; K < bs * bs; ++K) {
          // Reduce local sums within our group
          reduced_sums[K] = sycl::reduce_over_group(it.get_group(), sum[K], sycl::plus<T>());
        }

        // Write to scratch memory
        if (it.get_group().leader()) {
          for (unsigned int I = 0; I < bs; ++I)
            for (unsigned int J = 0; J < bs; ++J) {
              sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device> c_ref(c[I * bs + J]);
              c_ref += reduced_sums[I * bs + J];
            }
          // s[group_id * bs * bs + I * bs + J] = reduced_sums[I * bs + J];
        }
      });
    });
    if (events) events->push_back(reduce_event);

    // sycl::event finalize_event = q->single_task([=]() {
    //   for (unsigned int I = 0; I < bs; ++I)
    //     for (unsigned int J = 0; J < bs; ++J) c[I * bs + J] = 0;

    //   for (std::size_t i = 0; i < launch_.num_groups; ++i) {
    //     for (unsigned int I = 0; I < bs; ++I)
    //       for (unsigned int J = 0; J < bs; ++J) c[I * bs + J] += s[i * bs * bs + I * bs + J];
    //   }
    // });
    // if (events) events->push_back(finalize_event);
  }

  void subtract_product(PanelView Y, DenseMatrix<T>& M) { TRL_TODO("PanelView::subtract_product"); }

  T* data() { return start_; }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return count_ * bs; }

private:
  sycl::queue* q;

  T* start_;
  std::size_t rows_;
  unsigned int count_;

  T* scratch_;

  DotLaunch launch_;
};

} // namespace trl::Sycl
