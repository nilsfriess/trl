#pragma once

/** @file roofline.hh
 *  @brief Self-calibrating microbenchmarks for roofline numerators.
 *
 *  Instead of hardcoding theoretical peak bandwidth / FLOP numbers (which
 *  depend on the device, the SYCL backend, the thread count and even the
 *  access mix of the probe), this header measures *achievable* peaks through
 *  the very queue the benchmarked kernels run on:
 *
 *  - STREAM triad:   c[i] = a[i] + alpha * b[i], classic 2-read + 1-write
 *                    stream. Bytes are counted with the STREAM convention
 *                    (3 words per element; the write-allocate/RFO traffic a
 *                    CPU actually moves on the bus is not counted).
 *  - Copy stream:    c[i] = a[i], one read + one write. The yardstick for
 *                    kernels that stream one array in and one out, such as
 *                    the panel product W = V * M.
 *  - Read stream:    sum += a[i], a pure-read reduction. This is the honest
 *                    yardstick for read-only kernels such as the tall-skinny
 *                    dot (V^T W), whose access mix contains no writes.
 *  - FMA throughput: a register-resident FMA loop with enough independent
 *                    accumulators to cover FMA latency. Measures the FLOP/s
 *                    this backend/toolchain can extract from the device --
 *                    intentionally *not* the paper peak.
 *
 *  Two details decide whether these probes measure main memory at all:
 *
 *  - Access pattern. On GPUs the work-items of a sub-group must touch
 *    *consecutive* addresses or every load wastes most of its memory
 *    transaction; on CPUs the opposite is true (each thread wants a
 *    contiguous, prefetchable, vectorizable range). The probes therefore use
 *    an interleaved (grid-stride) loop on GPU devices and a contiguous-chunk
 *    loop everywhere else -- see detail::for_each_index().
 *  - Working set. The arrays must be several times larger than the last level
 *    cache, or the probe reports cache bandwidth. Modern GPUs have very large
 *    L2s (96 MB on a GB202), so the default size is derived from the device's
 *    reported cache size rather than fixed -- see detail::probe_bytes().
 *
 *  Partial reductions are combined per work-group and only the group leader
 *  updates the global sink; one device-scope atomic per work-item would
 *  serialize the whole probe and show up as (much) too little bandwidth.
 *
 *  All probes report the *minimum* kernel time over several repeats, measured
 *  with SYCL event profiling.
 *
 *  Requirements:
 *  - The queue must be created with sycl::property::queue::enable_profiling().
 *  - An in-order queue is assumed (kernels are sequenced by waiting on each
 *    event, so out-of-order queues work too, just with less overlap).
 *
 *  Caveats: numbers are "achievable for this backend and machine state";
 *  thermals / power management can make them drift, and on CPUs the result
 *  depends on the thread count the backend was launched with.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

#include <sycl/sycl.hpp>

#include "benchmark.hh"

namespace trl::benchmark {

/** @brief Achievable device peaks measured by measure_peaks(). */
struct DevicePeaks {
  double triad_gbps = 0.0; ///< STREAM triad bandwidth (3 words/element convention)
  double copy_gbps = 0.0;  ///< Copy bandwidth (1 read + 1 write per element)
  double read_gbps = 0.0;  ///< Pure-read stream bandwidth
  double fma_gflops = 0.0; ///< Achievable FMA throughput (mul+add = 2 flops)
};

namespace detail {

/** @brief min for std::size_t that works inside device code on all backends. */
inline std::size_t min_size(std::size_t a, std::size_t b) { return a < b ? a : b; }

/** @brief Default launch configuration: enough work-items to fill the device.
 *
 *  Work-items do not own a fixed amount of work (all probe loops are strided
 *  over the whole array), so the only requirement is to saturate the machine:
 *  a few resident waves per compute unit on GPUs, a handful of chunks per core
 *  on CPUs.
 */
inline sycl::nd_range<1> probe_range(const sycl::device& d)
{
  const std::size_t max_wg = d.get_info<sycl::info::device::max_work_group_size>();
  const std::size_t cus = std::max<std::size_t>(1, d.get_info<sycl::info::device::max_compute_units>());

  // GPU: several resident waves per compute unit. CPU: many more chunks than
  // cores, so the backend's work-item loop still has something to vectorize.
  const std::size_t local = std::min<std::size_t>(d.is_gpu() ? 256 : 128, max_wg);
  const std::size_t groups = d.is_gpu() ? 8 * cus : 128;
  return {groups * local, local};
}

/** @brief Size of one probe array, chosen so the working set misses the LLC.
 *
 *  Uses several times the device's reported cache size, clamped into a sane
 *  band and capped at a fraction of device memory (three of these are
 *  allocated at once, on top of whatever the caller already holds).
 */
inline std::size_t probe_bytes(const sycl::device& d)
{
  const std::size_t cache = d.get_info<sycl::info::device::global_mem_cache_size>();
  const std::size_t gmem = d.get_info<sycl::info::device::global_mem_size>();

  std::size_t bytes = std::max<std::size_t>(8 * cache, std::size_t{256} << 20);
  if (gmem > 0) bytes = std::min(bytes, gmem / 8);
  return std::max<std::size_t>(bytes, std::size_t{32} << 20);
}

/** @brief Runs a kernel-submitting callable warmups + repeats times and
 *  returns the minimum device-side kernel time in ms.
 */
template <class F>
double min_kernel_ms(int warmups, int repeats, F&& submit)
{
  for (int i = 0; i < warmups; ++i) submit().wait();

  // Note: finite sentinel, not infinity — under -ffast-math
  // (-ffinite-math-only) std::min(inf, x) misbehaves.
  double best = std::numeric_limits<double>::max();
  for (int i = 0; i < repeats; ++i) {
    auto e = submit(); // NOLINT: AdaptiveCpp's event::wait() is not const-qualified
    e.wait();
    best = std::min(best, kernel_ms(e));
  }
  return best;
}

/** @brief Calls @p f for every index of [0, n) owned by this work-item.
 *
 *  @p interleaved selects the access pattern: grid-stride (consecutive
 *  work-items touch consecutive elements, i.e. coalesced -- what a GPU wants)
 *  or one contiguous chunk per work-item (what a CPU core wants).
 */
template <class F>
inline void for_each_index(const sycl::nd_item<1>& it, std::size_t n, bool interleaved, F&& f)
{
  const std::size_t gid = it.get_global_linear_id();
  const std::size_t gsize = it.get_global_range().size();

  if (interleaved) {
    for (std::size_t i = gid; i < n; i += gsize) f(i);
  }
  else {
    const std::size_t chunk = (n + gsize - 1) / gsize;
    const std::size_t begin = min_size(gid * chunk, n);
    const std::size_t end = min_size(begin + chunk, n);
    for (std::size_t i = begin; i < end; ++i) f(i);
  }
}

/** @brief Adds the per-work-item @p value into @p sink with one atomic per group. */
template <class T>
inline void reduce_into(const sycl::nd_item<1>& it, T value, T* sink)
{
  const T group_sum = sycl::reduce_over_group(it.get_group(), value, sycl::plus<T>());
  if (it.get_group().leader()) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> sink_ref(*sink);
    sink_ref += group_sum;
  }
}

} // namespace detail

/** @brief Measures achievable bandwidth and FMA throughput on @p q's device.
 *
 *  @param q                       Queue with enable_profiling; all probes run here.
 *  @param stream_bytes_per_array  Size of each stream array (three are used).
 *                                 Must be several times larger than the last
 *                                 level cache so the probes hit main memory;
 *                                 0 (the default) derives it from the device.
 */
template <class T = double>
DevicePeaks measure_peaks(sycl::queue& q, std::size_t stream_bytes_per_array = 0)
{
  const auto dev = q.get_device();
  const auto range = detail::probe_range(dev);
  const bool interleaved = dev.is_gpu();
  if (stream_bytes_per_array == 0) stream_bytes_per_array = detail::probe_bytes(dev);
  const std::size_t n = stream_bytes_per_array / sizeof(T);

  T* a = sycl::malloc_device<T>(n, q);
  T* b = sycl::malloc_device<T>(n, q);
  T* c = sycl::malloc_device<T>(n, q);
  T* sink = sycl::malloc_device<T>(1, q);
  q.fill(a, static_cast<T>(1.0000000001), n);
  q.fill(b, static_cast<T>(0.9999999999), n);
  q.memset(c, 0, n * sizeof(T));
  q.wait();

  DevicePeaks peaks;
  std::cerr << "[peaks] alloc+init done (3 x " << (stream_bytes_per_array / (1 << 20)) << " MiB, " << range.get_global_range().size() / range.get_local_range().size() << " groups x "
            << range.get_local_range().size() << ", " << (interleaved ? "interleaved" : "chunked") << ")\n"
            << std::flush;

  // -------------------------------------------------------------------
  // STREAM triad: c[i] = a[i] + alpha * b[i]   (2 reads + 1 write)
  // -------------------------------------------------------------------
  {
    const T alpha = static_cast<T>(1.0000000001);
    auto submit = [&] {
      return q.submit(
          [&](sycl::handler& cgh) { cgh.parallel_for(range, [=](sycl::nd_item<1> it) { detail::for_each_index(it, n, interleaved, [=](std::size_t i) { c[i] = a[i] + alpha * b[i]; }); }); });
    };
    const double ms = detail::min_kernel_ms(2, 5, submit);
    const double bytes = 3.0 * static_cast<double>(n) * sizeof(T); // STREAM convention
    peaks.triad_gbps = bytes / (ms * 1e-3) / 1e9;
    std::cerr << "[peaks] triad done: " << peaks.triad_gbps << " GB/s\n" << std::flush;
  }

  // -------------------------------------------------------------------
  // Copy stream: c[i] = a[i]   (1 read + 1 write, as in W = V * M)
  // -------------------------------------------------------------------
  {
    auto submit = [&] {
      return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
          detail::for_each_index(it, n, interleaved, [=](std::size_t i) { c[i] = a[i]; });
        });
      });
    };
    const double ms = detail::min_kernel_ms(2, 5, submit);
    const double bytes = 2.0 * static_cast<double>(n) * sizeof(T);
    peaks.copy_gbps = bytes / (ms * 1e-3) / 1e9;
    std::cerr << "[peaks] copy done: " << peaks.copy_gbps << " GB/s\n" << std::flush;
  }

  // -------------------------------------------------------------------
  // Read stream: sum += a[i]   (pure read, as in dot / tall-skinny GEMM)
  // -------------------------------------------------------------------
  {
    q.memset(sink, 0, sizeof(T)).wait();
    auto submit = [&] {
      return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
          T sum = 0;
          detail::for_each_index(it, n, interleaved, [&](std::size_t i) { sum += a[i]; });
          detail::reduce_into(it, sum, sink);
        });
      });
    };
    const double ms = detail::min_kernel_ms(2, 5, submit);
    const double bytes = static_cast<double>(n) * sizeof(T);
    peaks.read_gbps = bytes / (ms * 1e-3) / 1e9;
    std::cerr << "[peaks] read done: " << peaks.read_gbps << " GB/s\n" << std::flush;
  }

  // -------------------------------------------------------------------
  // FMA throughput: register-resident FMA loop.
  //
  // Each work-item keeps kAcc x kVec independent accumulators to provide
  // enough instruction-level parallelism to cover the FMA latency (the
  // kVec-innermost lane loop vectorizes on CPU backends; on GPUs it unrolls
  // into extra scalar ILP). The multiplier depends on k so the compiler
  // cannot collapse the accumulator chain into a single multiply.
  // -------------------------------------------------------------------
  {
    constexpr int kVec = 4;
    constexpr int kAcc = 8;
    const std::size_t gsize = range.get_global_range().size();

    auto submit = [&](std::size_t iters) {
      return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
          const std::size_t gid = it.get_global_linear_id();

          T x[kVec], y[kAcc][kVec], acc[kAcc][kVec];
          for (int l = 0; l < kVec; ++l) x[l] = a[(gid * kVec + l) % n];
          for (int k = 0; k < kAcc; ++k)
            for (int l = 0; l < kVec; ++l) {
              y[k][l] = b[((gid + k + 1) * kVec + l) % n];
              acc[k][l] = 0;
            }

          for (std::size_t i = 0; i < iters; ++i)
            for (int k = 0; k < kAcc; ++k)
              for (int l = 0; l < kVec; ++l) acc[k][l] = sycl::fma(x[l], y[k][l], acc[k][l]);

          T sum = 0;
          for (int k = 0; k < kAcc; ++k)
            for (int l = 0; l < kVec; ++l) sum += acc[k][l];
          detail::reduce_into(it, sum, sink);
        });
      });
    };

    const double flops_per_iter = 2.0 * kVec * kAcc * static_cast<double>(gsize);

    // Calibrate the inner trip count for ~0.4 s of kernel time so the
    // measurement is resolution- and launch-overhead-insensitive on any
    // device, without taking forever on slow ones.
    constexpr std::size_t pilot_iters = 256;
    const double pilot_ms = detail::min_kernel_ms(0, 1, [&] { return submit(pilot_iters); });
    std::cerr << "[peaks] fma pilot: " << pilot_ms << " ms for " << pilot_iters << " iters\n" << std::flush;

    std::size_t iters;
    if (pilot_ms > 0.0) {
      const double target = pilot_iters * 400.0 / pilot_ms;
      // Clamp so a bad pilot cannot blow the run up.
      iters = static_cast<std::size_t>(std::min(target, 1e7));
      iters = std::max<std::size_t>(iters, pilot_iters);
    }
    else {
      iters = 1 << 16; // pilot unmeasurable; fixed fallback
    }
    std::cerr << "[peaks] fma calibrated iters = " << iters << "\n" << std::flush;

    const double ms = detail::min_kernel_ms(1, 3, [&] { return submit(iters); });
    peaks.fma_gflops = flops_per_iter * static_cast<double>(iters) / (ms * 1e-3) / 1e9;
    std::cerr << "[peaks] fma done: " << peaks.fma_gflops << " GFLOP/s\n" << std::flush;
  }

  // Consume the sink so the reductions cannot be elided.
  T host_sink;
  q.memcpy(&host_sink, sink, sizeof(T)).wait();
  (void)host_sink;

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
  sycl::free(sink, q);

  return peaks;
}

} // namespace trl::benchmark
