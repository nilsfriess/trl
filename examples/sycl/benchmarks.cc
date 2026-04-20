#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>
#include <trl/impl/sycl/multivector.hh>
#include <type_traits>

using namespace trl::sycl;

struct RooflineConfig {
  double stream_bandwidth_bytes_per_s;
  double peak_flops_per_s;
};

template <class Scalar>
constexpr double default_peak_flops_per_s()
{
  if constexpr (std::is_same_v<Scalar, float>) return 104.8e12;
  else if constexpr (std::is_same_v<Scalar, double>) return 1.637e12;
  else return 0.0;
}

template <class Scalar, unsigned int cols>
constexpr double arithmetic_intensity_flops_per_byte()
{
  return static_cast<double>(cols) / sizeof(Scalar);
}

// Implementation with grid stride loop and tiling
template <class T, std::size_t cols, bool transposed = false>
sycl::event dot3(sycl::queue* q, typename BlockMultivector<T, cols>::BlockView A,
                 typename BlockMultivector<T, cols>::BlockView B,
                 typename BlockMultivector<T, cols>::BlockMatrix::BlockView C)
{
  C.set_zero();

  constexpr auto TNTM = []() {
    if constexpr (cols == 1) return std::make_pair<unsigned int, unsigned int>(1, 1);
    else if constexpr (cols == 2) return std::make_pair<unsigned int, unsigned int>(2, 2);
    else if constexpr (cols == 4) return std::make_pair<unsigned int, unsigned int>(2, 2);
    else if constexpr (cols == 8) return std::make_pair<unsigned int, unsigned int>(4, 4);
    else if constexpr (cols >= 16) return std::make_pair<unsigned int, unsigned int>(4, 4);
    else return std::make_pair<unsigned int, unsigned int>(cols, 1);
  }();
  constexpr auto TN = std::get<0>(TNTM);
  constexpr auto TM = std::get<1>(TNTM);

  static_assert(cols % TN == 0);
  static_assert(cols % TM == 0);
  constexpr auto N = cols / TN;
  constexpr auto M = cols / TM;

  const auto ncus = q->get_device().get_info<sycl::info::device::max_compute_units>();
  const auto global_size = 128 * cols * ncus;

  constexpr auto m_index = [=](std::size_t midx, unsigned int tm) {
    if constexpr (transposed) return midx + tm * M;
    else return midx * TM + tm;
  };

  constexpr auto n_index = [=](std::size_t nidx, unsigned int tn) {
    if constexpr (transposed) return nidx + tn * N;
    else return nidx * TN + tn;
  };

  return q->submit([&](auto& cgh) {
    const auto* a = A.data;
    const auto* b = B.data;
    auto* c = C.data;
    const auto rows = A.rows();
    // const auto n = A.rows();
 
    cgh.parallel_for(sycl::range<1>{global_size}, [=](sycl::item<1> it) {
      const auto tid = it[0];

      const auto tile_id = tid % (N * M);
      const auto worker_id = tid / (N * M);
      const auto num_workers = global_size / (N * M);

      const auto midx = tile_id / N;
      const auto nidx = tile_id % N;

      T C_local[TM][TN] = {};

      for (std::size_t k = worker_id; k < rows; k += num_workers) {
        T a_frag[TM];
        T b_frag[TN];

        for (unsigned int tm = 0; tm < TM; ++tm) {
          const auto m = m_index(midx, tm);
          a_frag[tm] = a[to_linear_index(cols, k, m)];
        }
        for (unsigned int tn = 0; tn < TN; ++tn) {
          const auto n = n_index(nidx, tn);
          b_frag[tn] = b[to_linear_index(cols, k, n)];
        }

        for (unsigned int tm = 0; tm < TM; ++tm)
          for (unsigned int tn = 0; tn < TN; ++tn) C_local[tm][tn] += a_frag[tm] * b_frag[tn];
      }

      for (unsigned int tm = 0; tm < TM; ++tm)
        for (unsigned int tn = 0; tn < TN; ++tn) {
          const auto m = m_index(midx, tm);
          const auto n = n_index(nidx, tn);

          sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_c(
              c[to_linear_index(cols, m, n)]);
          atomic_c += C_local[tm][tn];
        }
    });
  });
}

// Same as dot3 but with "two-stage" reduction
template <class T, std::size_t cols, bool transposed = false>
sycl::event dot5(sycl::queue* q, typename BlockMultivector<T, cols>::BlockView A,
                 typename BlockMultivector<T, cols>::BlockView B,
                 typename BlockMultivector<T, cols>::BlockMatrix::BlockView C)
{
  C.set_zero();

  constexpr auto TNTM = []() {
    if constexpr (cols == 1) return std::make_pair<unsigned int, unsigned int>(1, 1);
    else if constexpr (cols == 2) return std::make_pair<unsigned int, unsigned int>(2, 2);
    else if constexpr (cols == 4) return std::make_pair<unsigned int, unsigned int>(2, 2);
    else if constexpr (cols == 8) return std::make_pair<unsigned int, unsigned int>(2, 2);
    else if constexpr (cols >= 16) return std::make_pair<unsigned int, unsigned int>(2, 2);
    else return std::make_pair<unsigned int, unsigned int>(cols, 1);
  }();
  constexpr auto TN = std::get<0>(TNTM);
  constexpr auto TM = std::get<1>(TNTM);

  static_assert(cols % TN == 0);
  static_assert(cols % TM == 0);
  constexpr auto N = cols / TN;
  constexpr auto M = cols / TM;

  const std::size_t local_size = 64;
  const auto ncus = q->get_device().get_info<sycl::info::device::max_compute_units>();
  const auto global_size = cols * ncus * local_size;

  constexpr auto m_index = [=](std::size_t midx, unsigned int tm) {
    if constexpr (transposed) return midx + tm * M;
    else return midx * TM + tm;
  };

  constexpr auto n_index = [=](std::size_t nidx, unsigned int tn) {
    if constexpr (transposed) return nidx + tn * N;
    else return nidx * TN + tn;
  };

  return q->submit([&](auto& cgh) {
    const T* __restrict__ a = A.data;
    const T* __restrict__ b = B.data;
    auto* c = C.data;
    const auto rows = A.rows();
    // const auto n = A.rows();

    sycl::local_accessor<T, 2> C_shared(sycl::range<2>(cols, cols), cgh);

    cgh.parallel_for(::sycl::nd_range<1>{global_size, local_size}, [=](::sycl::nd_item<1> it) {
      const auto tid = it.get_global_id(0);
      auto wg = it.get_group();

      const auto tile_id = tid % (N * M);
      const auto worker_id = tid / (N * M);
      const auto num_workers = it.get_global_range(0) / (N * M);

      const auto midx = tile_id / N;
      const auto nidx = tile_id % N;

      // Zero the shared memory
      if (wg.leader()) {
        for (unsigned int m = 0; m < cols; ++m)
          for (unsigned int n = 0; n < cols; ++n) C_shared[m][n] = 0;
      }
      wg.barrier();

      T C_local[TM][TN] = {};

      for (std::size_t k = worker_id; k < rows; k += num_workers) {
        T a_frag[TM];
        T b_frag[TN];

        for (unsigned int tm = 0; tm < TM; ++tm) {
          const auto m = m_index(midx, tm);
          a_frag[tm] = a[to_linear_index(cols, k, m)];
        }
        for (unsigned int tn = 0; tn < TN; ++tn) {
          const auto n = n_index(nidx, tn);
          b_frag[tn] = b[to_linear_index(cols, k, n)];
        }

        for (unsigned int tm = 0; tm < TM; ++tm)
          for (unsigned int tn = 0; tn < TN; ++tn) C_local[tm][tn] += a_frag[tm] * b_frag[tn];
      }

      for (unsigned int tm = 0; tm < TM; ++tm)
        for (unsigned int tn = 0; tn < TN; ++tn) {
          const auto m = m_index(midx, tm);
          const auto n = n_index(nidx, tn);

          ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed, ::sycl::memory_scope::work_group>
              atomic_c(C_shared[m][n]);
          atomic_c += C_local[tm][tn];
        }

      // Wait for everyone in the work-group to finish
      wg.barrier();

      if (wg.leader()) {
        for (unsigned int m = 0; m < cols; ++m)
          for (unsigned int n = 0; n < cols; ++n) {
            ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device>
                atomic_c(c[to_linear_index(cols, m, n)]);
            atomic_c += C_shared[m][n];
          }
      }
    });
  });
}

template <class T, std::size_t cols>
sycl::event dothost(sycl::queue* q, typename BlockMultivector<T, cols>::BlockView A,
                    typename BlockMultivector<T, cols>::BlockView B,
                    typename BlockMultivector<T, cols>::BlockMatrix::BlockView C)
{
  C.set_zero();

  const auto* data_ = A.data;
  const auto* bdata_ = B.data;
  const auto rows = A.rows();
  const auto upper = (rows / 4) * 4;

#pragma omp parallel
  {
    alignas(64) std::array<T, cols * cols> R_temp{};

#pragma omp for nowait
    for (std::size_t k = 0; k < upper; k += 4) {
      for (std::size_t i = 0; i < cols; ++i) {
        const T a0 = data_[(k + 0) * cols + i];
        const T a1 = data_[(k + 1) * cols + i];
        const T a2 = data_[(k + 2) * cols + i];
        const T a3 = data_[(k + 3) * cols + i];
#pragma omp simd
        for (std::size_t j = 0; j < cols; ++j)
          R_temp[i * cols + j] += a0 * bdata_[(k + 0) * cols + j] +
                                  a1 * bdata_[(k + 1) * cols + j] +
                                  a2 * bdata_[(k + 2) * cols + j] + a3 * bdata_[(k + 3) * cols + j];
      }
    }

    // Epilogue: remaining rows (at most 3), still parallel — one thread handles them
#pragma omp single nowait
    for (std::size_t k = upper; k < rows; ++k)
      for (std::size_t i = 0; i < cols; ++i) {
        const T a_ki = data_[k * cols + i];
#pragma omp simd
        for (std::size_t j = 0; j < cols; ++j) R_temp[i * cols + j] += a_ki * bdata_[k * cols + j];
      }

#pragma omp critical
    for (std::size_t i = 0; i < cols * cols; ++i) C.data[i] += R_temp[i];
  }

  return sycl::event{};
}

template <class Scalar, unsigned int cols, class DotFunc>
void run_benchmark(sycl::queue& q, std::size_t N, int runs, DotFunc dot,
                   const RooflineConfig& roofline)
{
  using BMV = BlockMultivector<Scalar, cols>;

  BMV A(q, N, cols);
  auto a = A.block_view(0);

  BMV B(q, N, cols);
  auto b = B.block_view(0);

  q.fill<Scalar>(a.data, Scalar{1.}, N * cols);
  q.fill<Scalar>(b.data, Scalar{2.}, N * cols);
  q.wait();

  // Verify correctness: run dot once on device, then compare with host result
  typename BMV::BlockMatrix C(q, 1, 1);
  auto c = C.block_view(0, 0);
  dot(&q, a, b, c);
  q.wait();

  Scalar C_ref[cols * cols] = {};
#pragma omp parallel for reduction(+ : C_ref[ : cols * cols])
  for (std::size_t row = 0; row < N; ++row)
    for (unsigned int i = 0; i < cols; ++i)
      for (unsigned int j = 0; j < cols; ++j)
        C_ref[i * cols + j] += a.data[row * cols + i] * b.data[row * cols + j];

  Scalar max_err = 0;
  for (unsigned int k = 0; k < cols * cols; ++k)
    max_err = std::max(max_err, std::abs(C_ref[k] - c.data[k]));

  if (max_err < 1e-6) std::cout << "Correctness check PASSED (max error: " << max_err << ")\n";
  else std::cout << "Correctness check FAILED (max error: " << max_err << ")\n";

  // Prefetch data to device
  q.prefetch(a.data, N * cols * sizeof(Scalar));
  q.prefetch(b.data, N * cols * sizeof(Scalar));
  q.prefetch(c.data, cols * cols * sizeof(Scalar));
  q.wait();

  unsigned long time = 0;
#if 0
  for (int i = 0; i < runs; ++i) {
    auto event = dot(&q, a, b, c);
    event.wait();
    auto end = event.template get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = event.template get_profiling_info<sycl::info::event_profiling::command_start>();
    time += end - start;
  }
#else
  for (int i = 0; i < runs; ++i) {
    q.wait();
    auto start = std::chrono::steady_clock::now();
    dot(&q, a, b, c);
    q.wait();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
#endif
  double avg_time = time / 1e9 / runs;
  double flops = 2.0 * N * cols * cols;
  double attained_flops_per_s = flops / avg_time;
  double gflops = flops / avg_time / 1e9;

  constexpr double arithmetic_intensity = arithmetic_intensity_flops_per_byte<Scalar, cols>();
  const double roofline_flops_per_s = std::min(
      roofline.peak_flops_per_s, arithmetic_intensity * roofline.stream_bandwidth_bytes_per_s);
  const double roofline_pct =
      roofline_flops_per_s > 0.0 ? 100.0 * attained_flops_per_s / roofline_flops_per_s : 0.0;
  const double ridge_point = roofline.stream_bandwidth_bytes_per_s > 0.0
                                 ? roofline.peak_flops_per_s / roofline.stream_bandwidth_bytes_per_s
                                 : 0.0;
  const bool is_compute_bound = arithmetic_intensity >= ridge_point;

  std::cout << "Avg time: " << avg_time << " s | " << gflops << " GFLOP/s\n";
  std::cout << "Algorithmic AI: " << arithmetic_intensity << " FLOP/byte"
            << " | Roofline bound: " << roofline_flops_per_s / 1e9 << " GFLOP/s" << " | "
            << roofline_pct << "% of roofline"
            << " | Regime: " << (is_compute_bound ? "compute-bound" : "memory-bound") << "\n";
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cout << "Usage: " << argv[0]
              << " <benchmark_number> [stream_bandwidth_MBps] [peak_tflops]\n";
    return 1;
  }
  int benchmark_number = std::stoi(argv[1]);

  using Scalar = double;

  sycl::queue q{{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};

  const double default_stream_bandwidth_mb_per_s = 1380007.896;
  const double stream_bandwidth_mb_per_s =
      argc >= 3 ? std::stod(argv[2]) : default_stream_bandwidth_mb_per_s;
  const double peak_tflops =
      argc >= 4 ? std::stod(argv[3]) : default_peak_flops_per_s<Scalar>() / 1e12;
  const RooflineConfig roofline{stream_bandwidth_mb_per_s * 1e6, peak_tflops * 1e12};

  int runs = 20;

  auto run_for_cols = [&]<unsigned int cols>() {
    constexpr std::size_t N = (cols >= 16 ? (cols >= 32 ? 1 << 25 : 1 << 26) : (1 << 27)) + 3;
    std::cout << "\n=== cols = " << cols << ", N = " << N << " ===\n";
    switch (benchmark_number) {
      case 1: {
        run_benchmark<Scalar, cols>(q, N, runs, dot3<Scalar, cols, false>, roofline);
        break;
      }
      case 2: {
        run_benchmark<Scalar, cols>(q, N, runs, dot3<Scalar, cols, true>, roofline);
        break;
      }
      case 3: {
        run_benchmark<Scalar, cols>(q, N, runs, dothost<Scalar, cols>, roofline);
        break;
      }
      case 4: {
        run_benchmark<Scalar, cols>(q, N, runs, dot5<Scalar, cols, true>, roofline);
        break;
      }
      default: std::cout << "Unknown benchmark number '" << benchmark_number << "'\n";
    }
  };

  run_for_cols.template operator()<2>();
  run_for_cols.template operator()<4>();
  run_for_cols.template operator()<8>();
  run_for_cols.template operator()<16>();
  run_for_cols.template operator()<32>();

  q.wait();
}
