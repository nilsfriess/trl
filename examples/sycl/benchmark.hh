#pragma once

/** @file benchmark.hh
 *  @brief Small reusable timing helpers for microbenchmarks.
 *
 *  Two timing modes are provided:
 *  - Wall-clock timing of a callable (via time_wall), including any queue
 *    waits the callable performs. This measures what the host experiences.
 *  - SYCL event profiling (kernel_ms), which reports the device-side execution
 *    time of individual kernels. Requires the queue to be constructed with
 *    sycl::property::queue::enable_profiling().
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

namespace trl::benchmark {

/** @brief Summary statistics for a set of per-iteration timings (in ms). */
struct Result {
  std::string name;
  std::size_t repeats = 0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double mean_ms = 0.0;
  double median_ms = 0.0;
  double stddev_ms = 0.0;
};

/** @brief Computes summary statistics for a list of per-iteration timings. */
inline Result summarize(std::string name, const std::vector<double>& times_ms)
{
  Result r;
  r.name = std::move(name);
  r.repeats = times_ms.size();
  if (times_ms.empty()) return r;

  auto sorted = times_ms;
  std::sort(sorted.begin(), sorted.end());

  r.min_ms = sorted.front();
  r.max_ms = sorted.back();
  const std::size_t mid = sorted.size() / 2;
  r.median_ms = sorted.size() % 2 == 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);

  const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
  r.mean_ms = sum / static_cast<double>(sorted.size());

  double sq_sum = 0.0;
  for (double t : sorted) sq_sum += (t - r.mean_ms) * (t - r.mean_ms);
  r.stddev_ms = std::sqrt(sq_sum / static_cast<double>(sorted.size()));

  return r;
}

/** @brief Prints a one-line summary of a Result. */
inline void report(const Result& r, std::ostream& os = std::cout)
{
  os << std::setw(36) << std::left << r.name << std::right << std::fixed << std::setprecision(3)
     << "  min " << std::setw(10) << r.min_ms << "  median " << std::setw(10) << r.median_ms << "  mean " << std::setw(10)
     << r.mean_ms << "  stddev " << std::setw(9) << r.stddev_ms << "  max " << std::setw(10) << r.max_ms << "  ms"
     << std::defaultfloat << "\n";
}

/** @brief Times @p f with std::chrono over @p repeats iterations after @p warmups warmup runs.
 *
 *  @p f is called once per iteration and should block until the measured work
 *  is done (e.g. by waiting on the SYCL queue) so that wall time matches the
 *  work being measured.
 */
template <std::invocable F>
Result time_wall(std::string name, int warmups, int repeats, F&& f)
{
  using clock = std::chrono::steady_clock;

  for (int i = 0; i < warmups; ++i) f();

  std::vector<double> times_ms;
  times_ms.reserve(static_cast<std::size_t>(repeats));
  for (int i = 0; i < repeats; ++i) {
    const auto start = clock::now();
    f();
    const auto stop = clock::now();
    times_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }

  return summarize(std::move(name), times_ms);
}

/** @brief Device-side execution time of a single SYCL kernel in ms.
 *
 *  Uses SYCL event profiling. The event's queue must have been created with
 *  sycl::property::queue::enable_profiling().
 */
inline double kernel_ms(const sycl::event& event)
{
  const auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1e-6;
}

/** @brief Summed device-side execution time of all kernels represented by @p events, in ms. */
inline double total_kernel_ms(const std::vector<sycl::event>& events)
{
  double total = 0.0;
  for (const auto& e : events) total += kernel_ms(e);
  return total;
}

} // namespace trl::benchmark
