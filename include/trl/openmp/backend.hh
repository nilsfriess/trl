#pragma once

#include "trl/concepts.hh"
#include "trl/impl/openmp/blockmatrix.hh"
#include "trl/impl/openmp/multivector.hh"

#include <algorithm>
#include <array>
#include <cstddef>

namespace trl::openmp {
/** @brief Stateless OpenMP backend.
 *
 *  Names the storage types, allocates them, and owns the host<->device
 *  boundary. On OpenMP the device is the host, so sync() is a no-op and
 *  to_host() is a plain copy.
 */
template <class T, unsigned int bs>
struct Backend {
  using Scalar = T;
  using Multivector = ::trl::openmp::BlockMultivector<T, bs>;
  using BlockMatrix = ::trl::openmp::BlockMatrix<T, bs>;
  static constexpr unsigned int blocksize = bs;

  /** @brief Host mirror of a block.
   *
   *  On OpenMP the device is the host, so the mirror aliases the block's own
   *  storage: there is nothing to copy and Access is irrelevant.
   */
  class HostMirror {
  public:
    HostMirror(Scalar* ptr, std::size_t size)
        : ptr(ptr)
        , size_(size)
    {
    }

    HostMirror(const HostMirror&) = delete;
    HostMirror(HostMirror&&) = delete;
    HostMirror& operator=(const HostMirror&) = delete;
    HostMirror& operator=(HostMirror&&) = delete;

    ~HostMirror() = default;

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
  BlockMatrix make_blockmatrix(unsigned int br, unsigned int bc) const { return {br, bc}; }

  void sync() const {}

  HostMirror host_block(BlockMatrix& M, [[maybe_unused]] Access access) const
  {
    const auto n_total = M.block_rows() * M.block_cols() * blocksize * blocksize;
    return {M.data, n_total};
  }

  HostMirror host_block(typename BlockMatrix::BlockView B, [[maybe_unused]] Access access) const { return {B.data_, bs * bs}; }

  HostMirror host_block(typename Multivector::BlockView V, [[maybe_unused]] Access access) const { return {V.data_, V.rows() * V.cols()}; }
};
} // namespace trl::openmp
