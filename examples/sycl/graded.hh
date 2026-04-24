#pragma once

#include "evp_base.hh"

#include <sycl/sycl.hpp>
#include <trl/impl/sycl/profiling.hh>

// Matrix-free "graded tridiagonal" EVP.
//
// The matrix is symmetric tridiagonal:
//
//   A[i,i]   = (N - i)      (diagonal decreases linearly from N down to 1)
//   A[i,i±1] = -1           (same off-diagonal coupling as the 1D Laplacian)
//
// The eigenvalues are approximately N, N-1, N-2, ..., 1 (with O(1) corrections
// from the off-diagonal terms), so consecutive eigenvalues near the top of the
// spectrum are separated by ~1. This is far better-conditioned than the 1D
// Laplacian, whose top eigenvalues cluster near 4 with gaps of O(π²/N²).
//
// Exact eigenvalues: not known analytically, so Spectra provides the reference.

template <class T, unsigned int bs>
class GradedTridiagEVP : public StandardEVPBase<T, bs> {
public:
  using Base = StandardEVPBase<T, bs>;

  GradedTridiagEVP(sycl::queue queue, typename Base::Index N)
      : Base(queue, N)
  {
  }

  void apply(typename Base::BlockView X, typename Base::BlockView Y)
  {
    T* X_data = X.data;
    T* Y_data = Y.data;
    const std::size_t N = this->N;

    static auto* ev = trl::sycl::SyclProfiler::get().registerOrGetEvent(trl::sycl::SyclProfiler::get().registerOrGetFamily("EVP"), "apply");

    auto e = this->queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
      auto i = idx[0];
      for (std::size_t j = 0; j < bs; ++j) {
        T diag = static_cast<T>(N - i); // N, N-1, ..., 1
        T val = diag * X_data[i * bs + j];
        if (i > 0) val -= X_data[(i - 1) * bs + j];
        if (i < N - 1) val -= X_data[(i + 1) * bs + j];
        Y_data[i * bs + j] = val;
      }
    });
    trl::sycl::SyclProfiler::get().pushEvent(ev, e);
  }
};
