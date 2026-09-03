#pragma once

#include "trl/concepts.hh"
#include "trl/impl/sycl/blockmatrix.hh"
#include "trl/impl/sycl/multivector.hh"

#include <algorithm>
#include <array>
#include <span>

namespace trl::Sycl {
/** @brief SYCL backend.
 *
 *  Names the storage types, allocates them, and owns the host<->device
 *  boundary. Device operations are enqueued to a sycl::queue, sync()
 *  calls wait() on the queue.
 */
template <class T, unsigned int bs>
struct Backend {
  using Scalar = T;
  using Multivector = BlockMultivector<T, bs>;
  using BlockMatrix = BlockMatrix<T, bs>;
  static constexpr unsigned int blocksize = bs;

  /** @brief Staged host mirror of a small bs x bs block.
   *
   *  Copies in on construction and/or back out on destruction according to
   *  @p access, so the device-side block is only current once the handle has
   *  gone out of scope.
   */
  template <std::size_t Extent = std::dynamic_extent>
  class HostBlock {
  public:
    HostBlock(sycl::queue queue, Access access, Scalar* device_ptr, std::size_t extent)
        : queue(queue)
        , access(access)
        , device_ptr(device_ptr)
    {
      if constexpr (Extent == std::dynamic_extent) host_vector.resize(extent);

      if (access == Access::Read or access == Access::ReadWrite) {
        if constexpr (Extent == std::dynamic_extent) queue.memcpy(host_vector.data(), device_ptr, sizeof(Scalar) * extent).wait();
        else queue.memcpy(host_array.data(), device_ptr, sizeof(Scalar) * Extent).wait();
      }
    }

    HostBlock(const HostBlock&) = delete;
    HostBlock(HostBlock&&) = delete;
    HostBlock& operator=(const HostBlock&) = delete;
    HostBlock& operator=(HostBlock&&) = delete;

    ~HostBlock()
    {
      if (access == Access::Write or access == Access::ReadWrite) {
        if constexpr (Extent == std::dynamic_extent) queue.memcpy(device_ptr, host_vector.data(), sizeof(Scalar) * host_vector.size()).wait();
        else queue.memcpy(device_ptr, host_array.data(), sizeof(Scalar) * Extent).wait();
      }
    }

    Scalar& operator[](std::size_t i)
    {
      if constexpr (Extent == std::dynamic_extent) return host_vector[i];
      else return host_array[i];
    }

    const Scalar& operator[](std::size_t i) const
    {
      if constexpr (Extent == std::dynamic_extent) return host_vector[i];
      else return host_array[i];
    }

    Scalar* data()
    {
      if constexpr (Extent == std::dynamic_extent) return host_vector.data();
      else return host_array.data();
    }

    const Scalar* data() const
    {
      if constexpr (Extent == std::dynamic_extent) return host_vector.data();
      else return host_array.data();
    }

    std::size_t size() const
    {
      if constexpr (Extent == std::dynamic_extent) return host_vector.size();
      else return host_array.size();
    }

  private:
    sycl::queue queue;
    Access access;
    Scalar* device_ptr;
    std::array<Scalar, bs * bs> host_array;
    std::vector<Scalar> host_vector;
  };

  explicit Backend(sycl::queue queue)
      : queue(queue)
  {
  }

  Multivector make_multivector(std::size_t n, unsigned int cols) const { return {queue, n, cols}; }
  BlockMatrix make_blockmatrix(unsigned int br, unsigned int bc) const { return {queue, br, bc}; }
  void sync() { queue.wait(); }

  HostBlock<std::dynamic_extent> host_block(BlockMatrix &M, Access access)
  {
    const auto n_total = M.block_rows() * M.block_cols() * blocksize * blocksize;
    return {queue, access, M.data(), n_total};
  }

  HostBlock<bs * bs> host_block(BlockMatrix::BlockView B, Access access)
  {
    // Here we know the size at compile time, so we can use the HostBlock
    // variant that does not allocate memory at runtime
    return {queue, access, B.data, bs * bs};
  }

  HostBlock<std::dynamic_extent> host_block(Multivector::BlockView V, Access access)
  {
    // Here we do not know the size at compile time, so we have to use the
    // dynamic_extent variant that does allocate memory at runtime.
    // This accessor is supposed to be only use for initialisation
    // anyway, so that should be fine.
    return {queue, access, V.data, V.rows() * V.cols()};
  }

private:
  sycl::queue queue;
};
} // namespace trl::Sycl
