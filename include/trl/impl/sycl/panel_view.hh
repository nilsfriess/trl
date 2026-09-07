#pragma once

#include "trl/concepts.hh"
#include "trl/helpers.hh"
#include "trl/impl/sycl/dense_matrix.hh"

#include <cstddef>
#include <hipSYCL/sycl/libkernel/memory.hpp>
#include <sycl/sycl.hpp>
#include <vector>

namespace trl::Sycl {
template <class T, unsigned int bs>
class PanelView {
public:
  static constexpr std::size_t dot_local_size = 128;
  static constexpr std::size_t dot_num_groups = 64;

  PanelView(sycl::queue* q_, T* start, std::size_t rows, unsigned int count, T* scratch)
      : q(q_)
      , start_(start)
      , rows_(rows)
      , count_(count)
      , scratch_(scratch)
  {
  }

  // void set_zero() { q->memset(start_, 0, rows_ * count_ * bs * sizeof(T)); } //

  void copy_from(PanelView other) { TRL_TODO("PanelView::copy_from"); }

  void mult(TransposeMode t, const DenseMatrix<T>& M, PanelView out) { TRL_TODO("PanelView::gemm"); }

  void subtract(PanelView other) { TRL_TODO("PanelView::subtract"); }

  /** @brief Computes out = this^T * Y.
   *
   *  Submits a two-kernel dot product (block reduction into scratch, followed
   *  by a final summation) to the in-order queue. If @p events is given, the
   *  events of all submitted kernels are appended to it, so callers can use
   *  event profiling to time the kernels individually.
   */
  void dot(PanelView Y, DenseMatrix<T>& out, std::vector<sycl::event>* events = nullptr)
  {
    TRL_CHECK(count_ == 1, "Only the single block panel case is implemented");

    constexpr auto local_size = dot_local_size;
    constexpr auto num_groups = dot_num_groups;
    constexpr auto global_size = num_groups * local_size;

    const T* a = data();
    const T* b = Y.data();
    T* c = out.data();
    T* s = scratch_;
    const auto n = rows();

    sycl::event memset_event = q->memset(c, 0, sizeof(T) * bs * bs);
    if (events) events->push_back(memset_event);

    sycl::event reduce_event = q->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> it) {
        auto gid = it.get_global_id();

        T sum[bs * bs] = {0};
        for (auto i = gid; i < n; i += it.get_global_range()) {
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
          auto group_id = it.get_group().get_group_id();
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

    //   for (std::size_t i = 0; i < num_groups; ++i) {
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
};

} // namespace trl::Sycl
