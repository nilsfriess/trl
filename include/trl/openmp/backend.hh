#pragma once

#include "trl/concepts.hh"
#include "trl/helpers.hh"
#include "trl/impl/openmp/dense_matrix.hh"
#include "trl/impl/openmp/multivector.hh"

#include <cstddef>

namespace trl::openmp {
/** @brief Stateless OpenMP backend.
 *
 *  Names the storage types, allocates them, and owns the host<->device
 *  boundary. On OpenMP the device is the host, so sync() is a no-op.
 */
template <class T, unsigned int bs>
struct Backend {
  using Scalar = T;
  using Multivector = BlockMultivector<T, bs>;
  using DenseMatrix = ::trl::openmp::DenseMatrix<T>;
  using OwnedDenseMatrix = ::trl::openmp::OwnedDenseMatrix<T>;
  static constexpr unsigned int blocksize = bs;

  /** @brief Host mirror of a matrix or a multivector block.
   *
   *  The storage is already host memory, so the mirror aliases it and Access
   *  is irrelevant: there is nothing to stage in or out.
   */
  class HostBlock {
  public:
    HostBlock(Scalar* ptr, std::size_t size)
        : ptr(ptr)
        , size_(size)
    {
    }

    HostBlock(const HostBlock&) = delete;
    HostBlock(HostBlock&&) = delete;
    HostBlock& operator=(const HostBlock&) = delete;
    HostBlock& operator=(HostBlock&&) = delete;

    ~HostBlock() = default;

    Scalar& operator[](std::size_t i) { return ptr[i]; }
    const Scalar& operator[](std::size_t i) const { return ptr[i]; }

    Scalar* data() { return ptr; }
    const Scalar* data() const { return ptr; }

    std::size_t size() const { return size_; }

  private:
    Scalar* ptr;
    std::size_t size_;
  };

  Multivector make_multivector(std::size_t n, unsigned int cols) const { return {n, cols}; }
  OwnedDenseMatrix make_dense_matrix(unsigned int rows, unsigned int cols) const { return {rows, cols}; }

  void sync() const {}

  HostBlock host_block(DenseMatrix M, [[maybe_unused]] Access access) const
  {
    TRL_CHECK(M.contiguous(), "host_block requires a contiguous matrix");
    return {M.data(), M.size()};
  }

  HostBlock host_block(typename Multivector::BlockView V, [[maybe_unused]] Access access) const { return {V.data(), V.rows() * V.cols()}; }
};
} // namespace trl::openmp
