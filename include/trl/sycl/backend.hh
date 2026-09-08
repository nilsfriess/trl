#pragma once

#include "trl/concepts.hh"
#include "trl/impl/sycl/dense_matrix.hh"
#include "trl/impl/sycl/multivector.hh"

#include <array>
#include <span>
#include <vector>

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
  using DenseMatrix = DenseMatrix<T>;
  using OwnedDenseMatrix = OwnedDenseMatrix<T>;
  static constexpr unsigned int blocksize = bs;

  /** @brief Staged host mirror of a device matrix.
   *
   *  Copies in on construction and/or back out on destruction according to
   *  @p access, so the device-side data is only current once the handle has
   *  gone out of scope. Each construction and destruction drains the queue, so
   *  these belong at restart boundaries, not inside the Lanczos step.
   */
  class HostBlock {
  public:
    HostBlock(sycl::queue queue, Access access, Scalar* device_ptr, std::size_t extent)
        : queue(queue)
        , access(access)
        , device_ptr(device_ptr)
        , host_vector(extent)
    {
      if (access == Access::Read or access == Access::ReadWrite) queue.memcpy(host_vector.data(), device_ptr, sizeof(Scalar) * extent).wait();
    }

    HostBlock(const HostBlock&) = delete;
    HostBlock(HostBlock&&) = delete;
    HostBlock& operator=(const HostBlock&) = delete;
    HostBlock& operator=(HostBlock&&) = delete;

    ~HostBlock()
    {
      if (access == Access::Write or access == Access::ReadWrite) queue.memcpy(device_ptr, host_vector.data(), sizeof(Scalar) * host_vector.size()).wait();
    }

    Scalar& operator[](std::size_t i) { return host_vector[i]; }
    const Scalar& operator[](std::size_t i) const { return host_vector[i]; }

    Scalar* data() { return host_vector.data(); }
    const Scalar* data() const { return host_vector.data(); }

    std::size_t size() const { return host_vector.size(); }

  private:
    sycl::queue queue;
    Access access;
    Scalar* device_ptr;
    std::vector<Scalar> host_vector;
  };

  explicit Backend(sycl::queue queue)
      : queue(queue)
  {
  }

  Multivector make_multivector(std::size_t n, unsigned int cols) const { return {queue, n, cols}; }
  OwnedDenseMatrix make_dense_matrix(unsigned int rows, unsigned int cols) const { return {queue, rows, cols}; }
  void sync() { queue.wait(); }

  HostBlock host_block(DenseMatrix M, Access access)
  {
    TRL_CHECK(M.contiguous(), "host_block requires a contiguous matrix");
    return {queue, access, M.data(), M.size()};
  }

  HostBlock host_block(Multivector::BlockView V, Access access) { return {queue, access, V.data(), V.rows() * V.cols()}; }

private:
  sycl::queue queue;
};
} // namespace trl::Sycl
