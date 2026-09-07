#pragma once

#include <algorithm>
#include <cstddef>

#include <sycl/sycl.hpp>

namespace trl::Sycl {

/** @brief Launch geometry and access pattern for the reduction kernels.
 *
 *  Produced once per BlockMultivector by dot_launch() and handed down to the
 *  views, so the object that sizes the scratch buffer is also the one that
 *  decides how many work-groups write into it.
 */
struct DotLaunch {
  std::size_t local_size = 128;
  std::size_t num_groups = 128;

  /** @brief Whether work-items stride through the rows (GPU) or own one
   *  contiguous chunk each (CPU). See PanelView::dot. */
  bool interleaved = false;

  std::size_t global_size() const { return local_size * num_groups; }
};

/** @brief Picks a launch geometry for @p dev and a panel of @p rows rows.
 *
 *  GPUs want a few resident waves per compute unit and coalesced loads
 *  (consecutive work-items on consecutive rows); CPUs want many more chunks
 *  than cores, each contiguous so it prefetches and vectorizes. The group
 *  count is only clamped down on panels too short to fill the device, where
 *  the group reduction and the final atomic would otherwise cost more than
 *  the sums themselves. A tall panel always gets the full grid: the kernel is
 *  memory bound, so occupancy is what hides the load latency.
 */
inline DotLaunch dot_launch(const sycl::device& dev, std::size_t rows)
{
  const std::size_t max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
  const std::size_t cus = std::max<std::size_t>(1, dev.get_info<sycl::info::device::max_compute_units>());
  const bool gpu = dev.is_gpu();

  DotLaunch cfg;
  cfg.local_size = std::min<std::size_t>(gpu ? 256 : 128, max_wg);
  cfg.num_groups = gpu ? 8 * cus : 128;
  cfg.interleaved = gpu;

  constexpr std::size_t min_rows_per_item = 8;
  const std::size_t wanted = rows / (min_rows_per_item * cfg.local_size);
  cfg.num_groups = std::clamp<std::size_t>(wanted, 1, cfg.num_groups);
  return cfg;
}

} // namespace trl::Sycl
