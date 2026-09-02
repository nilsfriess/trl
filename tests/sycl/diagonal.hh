#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "trl/common.hh"
#include "trl/sycl/backend.hh"

// Diagonal operator with entries diag[i] = (i + 1)^2.
// SYCL twin of tests/openmp/diagonal.hh; only the kernel launch differs.
template <class T, unsigned int bs>
class DiagonalEVPOperator : public trl::EuclideanDot<trl::Sycl::Backend<T, bs>> {
public:
  using BlockView = typename trl::Sycl::Backend<T, bs>::Multivector::BlockView;

  DiagonalEVPOperator(sycl::queue queue, std::size_t n)
      : queue(queue)
      , n_(n)
  {
    diag = sycl::malloc_shared<T>(n, queue);
    for (std::size_t i = 0; i < n; ++i) diag[i] = static_cast<T>((i + 1) * (i + 1));
  }

  ~DiagonalEVPOperator() { sycl::free(diag, queue); }

  DiagonalEVPOperator(const DiagonalEVPOperator&) = delete;
  DiagonalEVPOperator& operator=(const DiagonalEVPOperator&) = delete;

  void apply(BlockView X, BlockView Y)
  {
    // Hoisted into locals: a device lambda cannot capture `this`.
    const T* diag_ptr = diag;
    const T* X_data = X.data;
    T* Y_data = Y.data;

    queue.parallel_for(sycl::range<1>(n_), [=](sycl::id<1> idx) {
      const std::size_t k = idx[0];
      for (unsigned int i = 0; i < bs; ++i) Y_data[k * bs + i] = diag_ptr[k] * X_data[k * bs + i];
    });
  }

  std::size_t size() const { return n_; }

private:
  sycl::queue queue;
  std::size_t n_;
  T* diag = nullptr;
};
