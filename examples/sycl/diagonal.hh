#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <sycl/sycl.hpp>
#include <trl/impl/sycl/multivector.hh>

#include "evp_base.hh"

template <class T, unsigned int bs>
class DiagonalEVP : public StandardEVPBase<T, bs> {
public:
  using Base = StandardEVPBase<T, bs>;

  DiagonalEVP(sycl::queue queue, typename Base::Index N)
      : Base(queue, N)
  {
    // Diagonal matrix with entries: diag[i] = i+1
    diag = sycl::malloc_shared<T>(N, queue);
    for (typename Base::Index i = 0; i < N; ++i) diag[i] = static_cast<T>((i + 1) * (i + 1));
  }

  ~DiagonalEVP() { sycl::free(diag, this->queue); }

  void apply(typename Base::BlockView X, typename Base::BlockView Y)
  {
    // Y = D * X where D is diagonal
    const T* diag_ptr = diag;
    T* X_data = X.data;
    T* Y_data = Y.data;

    this->queue.parallel_for(sycl::range<1>(this->N), [=](sycl::id<1> idx) {
      auto i = idx[0];
      for (std::size_t j = 0; j < bs; ++j) Y_data[i * bs + j] = diag_ptr[i] * X_data[i * bs + j];
    });
  }

private:
  // Diagonal entries
  T* diag = nullptr;
};
