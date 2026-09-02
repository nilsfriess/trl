#pragma once

#include "trl/concepts.hh"
#include "trl/impl/sycl/blockmatrix.hh"
#include "trl/impl/sycl/multivector.hh"

#include <algorithm>
#include <array>

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
  class HostBlock {
  public:
    HostBlock(sycl::queue queue, Access access, BlockMatrix::BlockView B)
        : queue(queue)
        , access(access)
        , device_ptr(B.data)
    {
      if (access == Access::Read or access == Access::ReadWrite) queue.memcpy(host_data.data(), device_ptr, sizeof(Scalar) * bs * bs).wait();
    }

    HostBlock(const HostBlock&) = delete;
    HostBlock(HostBlock&&) = delete;
    HostBlock& operator=(const HostBlock&) = delete;
    HostBlock& operator=(HostBlock&&) = delete;

    ~HostBlock()
    {
      if (access == Access::Write or access == Access::ReadWrite) queue.memcpy(device_ptr, host_data.data(), sizeof(Scalar) * bs * bs).wait();
    }

    Scalar& operator[](std::size_t i) { return host_data[i]; }
    const Scalar& operator[](std::size_t i) const { return host_data[i]; }

    Scalar* data() { return host_data.data(); }
    const Scalar* data() const { return host_data.data(); }

    std::size_t size() const { return host_data.size(); }

  private:
    sycl::queue queue;
    Access access;
    Scalar* device_ptr;
    std::array<Scalar, bs * bs> host_data;
  };

  /** @brief Host mirror of a multivector block.
   *
   *  A multivector block is rows x bs and can be far too large to stage, but it
   *  lives in USM shared memory, so the mirror aliases it directly. The queue is
   *  drained on construction: without that, host reads and writes here would race
   *  against kernels still in flight.
   */
  class HostVectors {
  public:
    HostVectors(sycl::queue queue, [[maybe_unused]] Access access, Multivector::BlockView V)
        : ptr(V.data)
        , size_(V.rows() * V.cols())
    {
      queue.wait();
    }

    HostVectors(const HostVectors&) = delete;
    HostVectors(HostVectors&&) = delete;
    HostVectors& operator=(const HostVectors&) = delete;
    HostVectors& operator=(HostVectors&&) = delete;

    ~HostVectors() = default;

    Scalar& operator[](std::size_t i) { return ptr[i]; }
    const Scalar& operator[](std::size_t i) const { return ptr[i]; }

    Scalar* data() { return ptr; }
    const Scalar* data() const { return ptr; }

    std::size_t size() const { return size_; }

  private:
    Scalar* ptr;
    std::size_t size_;
  };

  explicit Backend(sycl::queue queue)
      : queue(queue)
  {
  }

  Multivector make_multivector(std::size_t n, unsigned int cols) const { return {queue, n, cols}; }
  BlockMatrix make_blockmatrix(unsigned int br, unsigned int bc) const { return {queue, br, bc}; }
  void sync() { queue.wait(); }

  HostBlock host_block(BlockMatrix::BlockView B, Access access) { return {queue, access, B}; }

  HostVectors host_block(Multivector::BlockView V, Access access) { return {queue, access, V}; }
private:
  sycl::queue queue;
};
} // namespace trl::Sycl
