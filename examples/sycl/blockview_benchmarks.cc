#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>
#include <trl/impl/sycl/multivector.hh>
#include <type_traits>
#include <vector>

using namespace trl::sycl;

namespace {

void print_usage(const char* program)
{
  std::cout << "Usage: " << program << " <benchmark_number> [stream_bandwidth_MBps] [peak_tflops] [runs] [cols]\n"
            << "\n"
            << "Benchmarks:\n"
            << "  1  BlockView::dot() using the SYCL backend default implementation\n"
            << "  2  BlockView::gemm<false>() with alpha = beta = 1\n"
            << "\n"
            << "Examples:\n"
            << "  " << program << " 1\n"
            << "  " << program << " 1 1380007.896 1.637 5\n"
            << "  " << program << " 1 1380007.896 1.637 1 8\n"
            << "  " << program << " 2 1380007.896 1.637 1 8\n";
}

bool is_help_flag(const std::string& argument) { return argument == "-h" || argument == "--help"; }

template <unsigned int cols>
bool matches_cols_filter(const std::optional<unsigned int>& cols_filter)
{
  return !cols_filter.has_value() || cols_filter.value() == cols;
}

} // namespace

struct RooflineConfig {
  double stream_bandwidth_bytes_per_s;
  double peak_flops_per_s;
};

struct BenchmarkMetrics {
  double flops;
  double arithmetic_intensity_flops_per_byte;
};

template <class Scalar>
constexpr double default_peak_flops_per_s()
{
  if constexpr (std::is_same_v<Scalar, float>) return 104.8e12;
  else if constexpr (std::is_same_v<Scalar, double>) return 1.637e12;
  else return 0.0;
}

template <class Scalar, unsigned int cols>
constexpr double dot_arithmetic_intensity_flops_per_byte()
{
  return static_cast<double>(cols) / sizeof(Scalar);
}

template <class Scalar, unsigned int cols>
constexpr BenchmarkMetrics dot_metrics(std::size_t rows)
{
  return BenchmarkMetrics{2.0 * static_cast<double>(rows) * cols * cols, dot_arithmetic_intensity_flops_per_byte<Scalar, cols>()};
}

template <class Scalar, unsigned int cols>
constexpr BenchmarkMetrics gemm_metrics(std::size_t rows)
{
  const double flops = 2.0 * static_cast<double>(rows) * cols * cols;
  const double bytes = static_cast<double>((3 * rows * cols + cols * cols) * sizeof(Scalar));
  return BenchmarkMetrics{flops, flops / bytes};
}

template <class Scalar, unsigned int cols>
std::array<Scalar, cols * cols> make_dot_reference(const Scalar* a_data, const Scalar* b_data, std::size_t rows)
{
  std::array<Scalar, cols * cols> reference{};

  for (std::size_t row = 0; row < rows; ++row)
    for (unsigned int i = 0; i < cols; ++i)
      for (unsigned int j = 0; j < cols; ++j) reference[i * cols + j] += a_data[row * cols + i] * b_data[row * cols + j];

  return reference;
}

template <class Scalar, std::size_t size>
Scalar max_abs_error(const Scalar* actual, const std::array<Scalar, size>& expected)
{
  Scalar max_err = 0;
  for (std::size_t index = 0; index < size; ++index) max_err = std::max(max_err, std::abs(expected[index] - actual[index]));
  return max_err;
}

template <class Scalar>
void print_correctness_result(Scalar max_err)
{
  if (max_err < 1e-6) std::cout << "Correctness check PASSED (max error: " << max_err << ")\n";
  else std::cout << "Correctness check FAILED (max error: " << max_err << ")\n";
}

template <class Scalar, unsigned int cols>
Scalar gemm_max_abs_error(const Scalar* a_data, const Scalar* b_data, const Scalar* c_data, std::size_t rows, Scalar initial_c_value)
{
  Scalar max_err = 0;

  for (std::size_t row = 0; row < rows; ++row)
    for (unsigned int i = 0; i < cols; ++i) {
      Scalar expected = initial_c_value;
      for (unsigned int j = 0; j < cols; ++j) expected += a_data[row * cols + j] * b_data[j * cols + i];
      max_err = std::max(max_err, std::abs(expected - c_data[row * cols + i]));
    }

  return max_err;
}

template <class Operation>
double time_operation(::sycl::queue& q, int runs, Operation&& operation)
{
  unsigned long long time_ns = 0;

  for (int run = 0; run < runs; ++run) {
    q.wait();
    auto start = std::chrono::steady_clock::now();
    operation();
    q.wait();
    auto end = std::chrono::steady_clock::now();
    time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  return static_cast<double>(time_ns) / 1e9 / runs;
}

void report_benchmark(double avg_time, const BenchmarkMetrics& metrics, const RooflineConfig& roofline)
{
  const double attained_flops_per_s = metrics.flops / avg_time;
  const double gflops = attained_flops_per_s / 1e9;
  const double roofline_flops_per_s =
      std::min(roofline.peak_flops_per_s, metrics.arithmetic_intensity_flops_per_byte * roofline.stream_bandwidth_bytes_per_s);
  const double roofline_pct = roofline_flops_per_s > 0.0 ? 100.0 * attained_flops_per_s / roofline_flops_per_s : 0.0;
  const double ridge_point = roofline.stream_bandwidth_bytes_per_s > 0.0 ? roofline.peak_flops_per_s / roofline.stream_bandwidth_bytes_per_s : 0.0;
  const bool is_compute_bound = metrics.arithmetic_intensity_flops_per_byte >= ridge_point;

  std::cout << "Avg time: " << avg_time << " s | " << gflops << " GFLOP/s\n";
  std::cout << "Algorithmic AI: " << metrics.arithmetic_intensity_flops_per_byte << " FLOP/byte"
            << " | Roofline bound: " << roofline_flops_per_s / 1e9 << " GFLOP/s" << " | " << roofline_pct << "% of roofline"
            << " | Regime: " << (is_compute_bound ? "compute-bound" : "memory-bound") << "\n";
}

template <class Scalar, unsigned int cols, class Operation, class ReferenceFactory>
void run_blockview_benchmark(::sycl::queue& q, std::size_t rows, int runs, Operation&& operation, ReferenceFactory&& make_reference,
                             const BenchmarkMetrics& metrics, const RooflineConfig& roofline)
{
  using BMV = BlockMultivector<Scalar, cols>;

  BMV A(q, rows, cols);
  auto a = A.block_view(0);

  BMV B(q, rows, cols);
  auto b = B.block_view(0);

  typename BMV::BlockMatrix C(q, 1, 1);
  auto c = C.block_view(0, 0);

  q.fill<Scalar>(a.data, Scalar{1.}, rows * cols);
  q.fill<Scalar>(b.data, Scalar{2.}, rows * cols);
  q.wait();

  operation(a, b, c);
  q.wait();

  std::vector<Scalar> a_host(rows * cols), b_host(rows * cols), c_host(cols * cols);
  q.memcpy(a_host.data(), a.data, rows * cols * sizeof(Scalar));
  q.memcpy(b_host.data(), b.data, rows * cols * sizeof(Scalar));
  q.memcpy(c_host.data(), c.data, cols * cols * sizeof(Scalar));
  q.wait();

  const auto reference = make_reference(a_host.data(), b_host.data(), rows);
  const Scalar max_err = max_abs_error(c_host.data(), reference);
  print_correctness_result(max_err);

  const double avg_time = time_operation(q, runs, [&]() { operation(a, b, c); });
  report_benchmark(avg_time, metrics, roofline);
}

template <class Scalar, unsigned int cols, class Operation, class ErrorComputer>
void run_multivector_output_benchmark(::sycl::queue& q, std::size_t rows, int runs, Operation&& operation, ErrorComputer&& compute_error,
                                      const BenchmarkMetrics& metrics, const RooflineConfig& roofline)
{
  using BMV = BlockMultivector<Scalar, cols>;

  BMV A(q, rows, cols);
  auto a = A.block_view(0);

  typename BMV::BlockMatrix B(q, 1, 1);
  auto b = B.block_view(0, 0);

  BMV C(q, rows, cols);
  auto c = C.block_view(0);

  constexpr Scalar a_fill_value{1.};
  constexpr Scalar b_fill_value{2.};
  constexpr Scalar c_fill_value{3.};

  q.fill<Scalar>(a.data, a_fill_value, rows * cols);
  q.fill<Scalar>(b.data, b_fill_value, cols * cols);
  q.fill<Scalar>(c.data, c_fill_value, rows * cols);
  q.wait();

  operation(a, b, c);
  q.wait();

  std::vector<Scalar> a_host(rows * cols), b_host(cols * cols), c_host(rows * cols);
  q.memcpy(a_host.data(), a.data, rows * cols * sizeof(Scalar));
  q.memcpy(b_host.data(), b.data, cols * cols * sizeof(Scalar));
  q.memcpy(c_host.data(), c.data, rows * cols * sizeof(Scalar));
  q.wait();

  const Scalar max_err = compute_error(a_host.data(), b_host.data(), c_host.data(), rows, c_fill_value);
  print_correctness_result(max_err);

  const double avg_time = time_operation(q, runs, [&]() { operation(a, b, c); });
  report_benchmark(avg_time, metrics, roofline);
}

template <class Scalar, unsigned int cols>
void run_dot_benchmark(::sycl::queue& q, std::size_t rows, int runs, const RooflineConfig& roofline)
{
  const auto metrics = dot_metrics<Scalar, cols>(rows);

  run_blockview_benchmark<Scalar, cols>(
      q, rows, runs, [](auto a, auto b, auto c) { a.dot(b, c); },
      [](const Scalar* a_data, const Scalar* b_data, std::size_t rows_count) { return make_dot_reference<Scalar, cols>(a_data, b_data, rows_count); },
      metrics, roofline);
}

template <class Scalar, unsigned int cols>
void run_gemm_benchmark(::sycl::queue& q, std::size_t rows, int runs, const RooflineConfig& roofline)
{
  const auto metrics = gemm_metrics<Scalar, cols>(rows);

  run_multivector_output_benchmark<Scalar, cols>(
      q, rows, runs, [](auto a, auto b, auto c) { c.template gemm<false>(Scalar{1.}, a, b, Scalar{1.}); },
      [](const Scalar* a_data, const Scalar* b_data, const Scalar* c_data, std::size_t rows_count, Scalar initial_c_value) {
        return gemm_max_abs_error<Scalar, cols>(a_data, b_data, c_data, rows_count, initial_c_value);
      },
      metrics, roofline);
}

int main(int argc, char* argv[])
{
  if (argc < 2 || is_help_flag(argv[1])) {
    print_usage(argv[0]);
    return argc < 2 ? 1 : 0;
  }

  const int benchmark_number = std::stoi(argv[1]);
  using Scalar = double;

  ::sycl::queue q{{::sycl::property::queue::in_order{}, ::sycl::property::queue::enable_profiling{}}};

  const double default_stream_bandwidth_mb_per_s = 1380007.896;
  const double stream_bandwidth_mb_per_s = argc >= 3 ? std::stod(argv[2]) : default_stream_bandwidth_mb_per_s;
  const double peak_tflops = argc >= 4 ? std::stod(argv[3]) : default_peak_flops_per_s<Scalar>() / 1e12;
  const int runs = argc >= 5 ? std::stoi(argv[4]) : 20;
  const std::optional<unsigned int> cols_filter = argc >= 6 ? std::optional<unsigned int>(std::stoul(argv[5])) : std::nullopt;
  const RooflineConfig roofline{stream_bandwidth_mb_per_s * 1e6, peak_tflops * 1e12};

  auto run_for_cols = [&]<unsigned int cols>() {
    if (!matches_cols_filter<cols>(cols_filter)) return;

    constexpr std::size_t rows = 174080;
    std::cout << "\n=== cols = " << cols << ", N = " << rows << " ===\n";

    switch (benchmark_number) {
      case 1: run_dot_benchmark<Scalar, cols>(q, rows, runs, roofline); break;
      case 2: run_gemm_benchmark<Scalar, cols>(q, rows, runs, roofline); break;
      default: std::cout << "Unknown benchmark number '" << benchmark_number << "'\n"; break;
    }
  };

  run_for_cols.template operator()<2>();
  run_for_cols.template operator()<4>();
  run_for_cols.template operator()<8>();
  run_for_cols.template operator()<16>();
  run_for_cols.template operator()<32>();

  q.wait();

  trl::sycl::SyclProfiler::get().report();
}
