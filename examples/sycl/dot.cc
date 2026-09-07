#include "trl/concepts.hh"
#include "trl/impl/sycl/multivector.hh"
#include "trl/sycl/backend.hh"

#include "benchmark.hh"

#include <Eigen/Core>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

using namespace trl::Sycl;
using trl::benchmark::report;
using trl::benchmark::Result;
using trl::benchmark::summarize;
using trl::benchmark::time_wall;
using trl::benchmark::total_kernel_ms;

namespace {

template <class Scalar, unsigned int blocksize>
Eigen::Matrix<Scalar, blocksize, blocksize> eigen_dot(const Scalar* Vh, const Scalar* Wh, std::size_t rows)
{
  constexpr int options = blocksize == 1 ? Eigen::ColMajor : Eigen::RowMajor;

  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, blocksize, options>> Vm(const_cast<Scalar*>(Vh), rows, blocksize);
  Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, blocksize, options>> Wm(const_cast<Scalar*>(Wh), rows, blocksize);

  // C = V^T * W, i.e. C(I, J) = <V_i, W_j> summed over the rows of the block.
  // The SYCL dot kernel stores C in row-major layout, so M_host[i * bs + j]
  // corresponds to C(i, j); Eigen's operator() handles the storage order.
  return Vm.transpose() * Wm;
}

} // namespace

int main()
{
  // enable_profiling is required for sycl event profiling of the kernels.
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};

  using Scalar = double;
  constexpr unsigned int blocksize = 4;
  const std::size_t n = 1 << 16;

  constexpr int warmups = 50;
  constexpr int repeats = 1000;

  BlockMultivector<Scalar, blocksize> V(q, n, 2 * blocksize);

  std::mt19937 rng;
  std::normal_distribution<Scalar> dist;

  Backend<Scalar, blocksize> backend(q);

  {
    auto host = backend.host_block(V.block_view(0), trl::Access::Write);
    std::generate_n(host.data(), host.size(), [&]() { return dist(rng); });
  }

  {
    auto host = backend.host_block(V.block_view(1), trl::Access::Write);
    std::generate_n(host.data(), host.size(), [&]() { return dist(rng); });
  }

  // ---------------------------------------------------------------------
  // Correctness check: SYCL dot vs. Eigen reference on the host.
  // ---------------------------------------------------------------------
  bool correct = true;
  {
    std::cout << "Correctness check (n = " << n << "):\n";

    const auto Vh = backend.host_block(V.block_view(0), trl::Access::Read);
    const auto Wh = backend.host_block(V.block_view(1), trl::Access::Read);
    const Eigen::Matrix<Scalar, blocksize, blocksize> ref_dot = eigen_dot<Scalar, blocksize>(Vh.data(), Wh.data(), n);

    DenseMatrix<Scalar> M(q, blocksize, blocksize);
    V.block_view(0).dot(V.block_view(1), M);
    q.wait();

    auto M_host = backend.host_block(M, trl::Access::Read);
    for (unsigned int i = 0; i < blocksize; ++i) {
      for (unsigned int j = 0; j < blocksize; ++j) {
        std::cout << "(" << M_host[i * blocksize + j] << " | " << ref_dot(i, j) << ")    ";
        if (std::abs((M_host[i * blocksize + j] - ref_dot(i, j)) / ref_dot(i, j)) > 1e-10) correct = false;
      }
      std::cout << "\n";
    }

    std::cout << "Result is " << (correct ? "correct" : "wrong") << "\n\n";
    if (!correct) return EXIT_FAILURE;
  }

  // ---------------------------------------------------------------------
  // Benchmark: Eigen on the host.
  // ---------------------------------------------------------------------
  const auto Vh = backend.host_block(V.block_view(0), trl::Access::Read);
  const auto Wh = backend.host_block(V.block_view(1), trl::Access::Read);

  volatile Scalar sink = 0; // accumulate results so the compiler cannot elide the work
  Result eigen_result = time_wall("Eigen dot (host wall time)", warmups, repeats, [&] {
    Eigen::Matrix<Scalar, blocksize, blocksize> res = eigen_dot<Scalar, blocksize>(Vh.data(), Wh.data(), n);
    sink += res(0, 0);
  });
  (void)sink;

  // ---------------------------------------------------------------------
  // Benchmark: SYCL kernels, timed with host wall time and SYCL event profiling.
  // ---------------------------------------------------------------------
  DenseMatrix<Scalar> M(q, blocksize, blocksize);
  std::vector<sycl::event> events;
  std::vector<double> kernel_times_ms;
  kernel_times_ms.reserve(static_cast<std::size_t>(repeats));

  Result sycl_wall_result = time_wall("SYCL dot (host wall time)", warmups, repeats, [&] {
    events.clear();
    V.block_view(0).dot(V.block_view(1), M, &events);
    q.wait();
    kernel_times_ms.push_back(total_kernel_ms(events));
  });

  Result sycl_kernel_result = summarize("SYCL dot (event profiling)", kernel_times_ms);

  // ---------------------------------------------------------------------
  // Report.
  // ---------------------------------------------------------------------
  std::cout << "Benchmark results (" << repeats << " repeats, " << warmups << " warmups, n = " << n << "):\n";
  report(eigen_result);
  report(sycl_wall_result);
  report(sycl_kernel_result);

  const double eigen_min = eigen_result.min_ms;
  const double sycl_wall_min = sycl_wall_result.min_ms;
  const double sycl_kernel_min = sycl_kernel_result.min_ms;
  std::cout << "\nMin-based comparison:\n";
  std::cout << "  Eigen / SYCL (wall time):        " << (sycl_wall_min > 0 ? eigen_min / sycl_wall_min : 0.0) << "x\n";
  std::cout << "  Eigen / SYCL (kernel time only): " << (sycl_kernel_min > 0 ? eigen_min / sycl_kernel_min : 0.0) << "x\n";
}
