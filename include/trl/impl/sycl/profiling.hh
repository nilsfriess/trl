#pragma once

// profiling.hh — SYCL backend profiling singleton.
//
// Usage:
//   1. Build with CMake option TRL_ENABLE_SYCL_PROFILING=ON (adds -DTRL_ENABLE_SYCL_PROFILING).
//   2. Create the SYCL queue with sycl::property::queue::enable_profiling{}.
//   3. In user code, call SyclProfiler::get().report(std::cout) after the work is done.
//
// Design:
//   - Singleton; thread-safe registration and event pushing.
//   - Events are stored lazily and queried (waited on) at report() time.
//   - Each operation uses a function-local static Event* initialised exactly once;
//     no per-call string lookup in the hot path.
//   - When TRL_ENABLE_SYCL_PROFILING is NOT defined the entire class compiles to
//     zero overhead (all methods are no-ops inlined away by the compiler).

#include <sycl/sycl.hpp>

#ifdef TRL_ENABLE_SYCL_PROFILING

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <list>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

namespace trl::sycl {

/**
 * @brief Singleton profiler for the SYCL backend.
 *
 * Typical use in a backend class:
 * @code
 * // Declare once per logical operation (e.g. at the top of the method):
 * static auto* ev = SyclProfiler::get().registerOrGetEvent(
 *     SyclProfiler::get().registerOrGetFamily("BlockView"), "dot");
 *
 * // After submitting to the queue, push the returned sycl::event:
 * auto sycl_e = q->submit(...);
 * SyclProfiler::get().pushEvent(ev, sycl_e);
 * @endcode
 *
 * At program end (or whenever desired):
 * @code
 * SyclProfiler::get().report(std::cout);
 * @endcode
 *
 * @note The queue must have been created with
 *       @c sycl::property::queue::enable_profiling{} — otherwise report() will
 *       print a warning for each event that cannot be queried.
 */
class SyclProfiler {
public:
  static SyclProfiler& get()
  {
    static SyclProfiler instance;
    return instance;
  }

  SyclProfiler() = default;
  SyclProfiler(const SyclProfiler&) = delete;
  SyclProfiler(SyclProfiler&&) = delete;
  SyclProfiler& operator=(const SyclProfiler&) = delete;
  SyclProfiler& operator=(SyclProfiler&&) = delete;
  ~SyclProfiler() = default;

  // -------------------------------------------------------------------
  // Data structures
  // -------------------------------------------------------------------

  /** @brief Accumulated timing data for one named operation. */
  struct Event {
    std::string name;
    std::size_t times_called{0};
    std::uint64_t total_ns{0};
    std::vector<::sycl::event> pending_events;
    std::mutex mtx; // protects pending_events, times_called
  };

  /** @brief A named group of events (e.g. "BlockView"). */
  struct Family {
    std::string name;
    std::list<Event> events; // std::list for pointer/reference stability
                             // (Event contains a non-movable std::mutex)
  };

  // -------------------------------------------------------------------
  // Registration API  (called during static initialisation, rarely)
  // -------------------------------------------------------------------

  /** Register or retrieve a family by name. Thread-safe. */
  Family* registerOrGetFamily(const std::string& name)
  {
    std::lock_guard lock(registry_mutex_);
    for (auto& f : families_)
      if (f.name == name) return &f;
    return &families_.emplace_back(Family{name, {}});
  }

  /**
   * Register or retrieve a named event within a family. Thread-safe.
   *
   * The returned pointer remains valid for the lifetime of the profiler
   * (families and events are stored in std::list, which never invalidates
   * existing iterators/pointers on insertion).
   */
  Event* registerOrGetEvent(Family* family, const std::string& event_name)
  {
    std::lock_guard lock(registry_mutex_);
    for (auto& e : family->events)
      if (e.name == event_name) return &e;
    auto& e = family->events.emplace_back();
    e.name = event_name;
    return &e;
  }

  // -------------------------------------------------------------------
  // Hot path: push a sycl::event to an Event slot
  // -------------------------------------------------------------------

  /**
   * Store a SYCL event for later profiling.  Called after every queue
   * submission.  Only acquires the per-event mutex (not the registry mutex).
   */
  void pushEvent(Event* event, ::sycl::event sycl_event)
  {
    std::lock_guard lock(event->mtx);
    event->pending_events.push_back(std::move(sycl_event));
    ++event->times_called;
  }

  // -------------------------------------------------------------------
  // Reporting
  // -------------------------------------------------------------------

  /**
   * Wait for all pending events, accumulate their execution times, and
   * print a formatted timing table to @p out.
   *
   * If the queue was not created with
   * @c sycl::property::queue::enable_profiling{} a single warning line is
   * printed instead of crashing.
   */
  void report(std::ostream& out = std::cout)
  {
    bool warned = false;

    // Drain pending sycl::events for a single Event slot.
    auto drain = [&](Event& ev) {
      std::lock_guard lock(ev.mtx);
      for (auto& se : ev.pending_events) {
        try {
          se.wait();
          const auto start = se.get_profiling_info<::sycl::info::event_profiling::command_start>();
          const auto end   = se.get_profiling_info<::sycl::info::event_profiling::command_end>();
          if (end > start) ev.total_ns += end - start;
        } catch (const ::sycl::exception& ex) {
          if (!warned) {
            out << "[SyclProfiler] Warning: could not query profiling info.\n"
                << "  Make sure the queue was created with "
                   "sycl::property::queue::enable_profiling{}.\n"
                << "  SYCL error: " << ex.what() << "\n\n";
            warned = true;
          }
        }
      }
      ev.pending_events.clear();
    };

    constexpr int W_EVENT  = 22;
    constexpr int W_TOTAL  = 17;
    constexpr int W_MEAN   = 16;
    constexpr int W_CALLS  = 12;
    const std::string sep(W_EVENT + 2 + W_TOTAL + 3 + W_MEAN + 3 + W_CALLS, '-');

    out << "\n==========================================================================================\n";
    out << "#                              SYCL Profiling Report                                     #\n";
    out << "==========================================================================================\n";
    out << std::left  << std::setw(W_EVENT) << "Event"
        << "| " << std::right << std::setw(W_TOTAL) << "Total time [ms]"
        << " | "  << std::right << std::setw(W_MEAN)  << "Mean time [ms]"
        << " | "  << std::right << std::setw(W_CALLS) << "Times called"
        << "\n" << sep << "\n";

    for (auto& family : families_) {
      out << std::left << std::setw(W_EVENT) << family.name << "|\n";
      for (auto& ev : family.events) {
        drain(ev);
        const double total_ms = static_cast<double>(ev.total_ns) * 1e-6;
        const double mean_ms  = ev.times_called > 0
                                    ? total_ms / static_cast<double>(ev.times_called)
                                    : 0.0;
        out << "  " << std::left  << std::setw(W_EVENT - 2) << ev.name
            << "| " << std::right << std::fixed << std::setprecision(3)
                                  << std::setw(W_TOTAL) << total_ms
            << " | " << std::right << std::fixed << std::setprecision(3)
                                   << std::setw(W_MEAN)  << mean_ms
            << " | " << std::right << std::setw(W_CALLS) << ev.times_called
            << "\n";
      }
    }
    out << sep << "\n";
  }

private:
  std::mutex registry_mutex_;
  std::list<Family> families_; // std::list for pointer stability
};

} // namespace trl::sycl

#else // ---- stub (TRL_ENABLE_SYCL_PROFILING not defined) ----------------------

#include <iostream>
#include <ostream>
#include <string>

namespace trl::sycl {

/**
 * @brief No-op stub for SyclProfiler used when TRL_ENABLE_SYCL_PROFILING is off.
 *
 * All methods are empty and will be completely eliminated by the compiler.
 */
class SyclProfiler {
public:
  static SyclProfiler& get()
  {
    static SyclProfiler instance;
    return instance;
  }

  SyclProfiler() = default;
  SyclProfiler(const SyclProfiler&) = delete;
  SyclProfiler(SyclProfiler&&) = delete;
  SyclProfiler& operator=(const SyclProfiler&) = delete;
  SyclProfiler& operator=(SyclProfiler&&) = delete;
  ~SyclProfiler() = default;

  struct Event {};
  struct Family {};

  Family* registerOrGetFamily(const std::string& /*name*/) { return nullptr; }
  Event*  registerOrGetEvent(Family* /*family*/, const std::string& /*name*/) { return nullptr; }
  void    pushEvent(Event* /*event*/, ::sycl::event /*sycl_event*/) {}
  void    report(std::ostream& /*out*/ = std::cout) {}
};

} // namespace trl::sycl

#endif // TRL_ENABLE_SYCL_PROFILING
