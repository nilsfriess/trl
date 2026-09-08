/** @file mult.cc
 *  @brief Benchmark and roofline analysis for the panel product W = V * M.
 *
 *  Companion to dot.cc. Where dot measures the tall-skinny reduction V^T W,
 *  this measures the tall-skinny GEMM behind PERFORMANCE.md pattern 4 (the
 *  restart Ritz back-transform): V is n x ncv, M is ncv x nev, W is n x nev.
 *
 *  Both TransposeMode::NoTranspose and TransposeMode::Transpose are checked
 *  for correctness against the Eigen reference; only NoTranspose is timed.
 *
 *  Panel layout: a panel of `count` blocks is `count` consecutive n x bs
 *  row-major blocks, *not* one n x (count * bs) row-major array. Column q of
 *  the panel therefore lives at data[(q / bs) * n * bs + i * bs + (q % bs)].
 *  The Eigen reference below assembles a dense n x p mirror accordingly.
 *
 *  Usage: mult_sycl [n] [ncv] [nev] [repeats] [warmups]
 */

#include "trl/concepts.hh"
#include "trl/impl/sycl/multivector.hh"
#include "trl/sycl/backend.hh"

#include "benchmark.hh"
#include "roofline.hh"

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

using namespace trl::Sycl;
using trl::benchmark::report;
using trl::benchmark::Result;
using trl::benchmark::summarize;
using trl::benchmark::time_wall;
using trl::benchmark::total_kernel_ms;

namespace {

template <class Scalar>
using RowMajorMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/** @brief Dense n x (blocks * bs) host mirror of a panel.
 *
 *  Undoes the block-of-blocks layout: block j of the panel supplies columns
 *  [j * bs, (j + 1) * bs) of the result.
 */
template <class Scalar, unsigned int bs>
RowMajorMatrix<Scalar> gather_panel(const Scalar* panel, std::size_t rows, unsigned int blocks)
{
  RowMajorMatrix<Scalar> out(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(blocks * bs));

  for (unsigned int j = 0; j < blocks; ++j) {
    // A single block is contiguous and row-major, so it maps directly.
    Eigen::Map<const RowMajorMatrix<Scalar>> block(panel + static_cast<std::size_t>(j) * rows * bs, static_cast<Eigen::Index>(rows), bs);
    out.block(0, static_cast<Eigen::Index>(j) * bs, static_cast<Eigen::Index>(rows), bs) = block;
  }

  return out;
}

/** @brief Parses argv[i] as a positive size, or returns @p fallback. */
std::size_t arg_or(int argc, char** argv, int i, std::size_t fallback)
{
  if (argc <= i) return fallback;
  const auto value = std::stoull(argv[i]);
  return value > 0 ? static_cast<std::size_t>(value) : fallback;
}

} // namespace

int main(int argc, char** argv)
{
  using Scalar = double;
  constexpr unsigned int blocksize = 4;

  // Defaults: the restart shape of PERFORMANCE.md pattern 4 (ncv = 64,
  // nev = 16) at an n large enough to leave every cache behind on both a CPU
  // and a GPU, without the 6.4 GB panel that dot.cc's n would imply here.
  const std::size_t n = arg_or(argc, argv, 1, std::size_t{1} << 21);
  const auto ncv = static_cast<unsigned int>(arg_or(argc, argv, 2, 64));
  const auto nev = static_cast<unsigned int>(arg_or(argc, argv, 3, 16));
  const auto repeats = static_cast<int>(arg_or(argc, argv, 4, 20));
  const auto warmups = static_cast<int>(arg_or(argc, argv, 5, 10));

  if (ncv % blocksize != 0 or nev % blocksize != 0) {
    std::cerr << "ncv and nev must both be multiples of the blocksize (" << blocksize << ")\n";
    return EXIT_FAILURE;
  }
  if (nev >= ncv) {
    std::cerr << "nev must be strictly smaller than ncv\n";
    return EXIT_FAILURE;
  }

  const unsigned int in_blocks = ncv / blocksize;   // columns of V, in blocks
  const unsigned int out_blocks = nev / blocksize;  // columns of W, in blocks

  // enable_profiling is required for sycl event profiling of the kernels.
  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};

  BlockMultivector<Scalar, blocksize> V(q, n, ncv);
  BlockMultivector<Scalar, blocksize> W(q, n, nev);
  OwnedDenseMatrix<Scalar> M(q, ncv, nev);

  Backend<Scalar, blocksize> backend(q);

  std::mt19937 rng;
  std::normal_distribution<Scalar> dist;

  {
    auto host = backend.host_block(V.panel_view(0, in_blocks), trl::Access::Write);
    std::generate_n(host.data(), host.size(), [&]() { return dist(rng); });
  }

  {
    auto host = backend.host_block(M, trl::Access::Write);
    std::generate_n(host.data(), host.size(), [&]() { return dist(rng); });
  }

  std::cout << "W = V * M  (n = " << n << ", V: " << n << " x " << ncv << ", M: " << ncv << " x " << nev << ", bs = " << blocksize << ")\n";
  std::cout << "  V " << (static_cast<double>(n) * ncv * sizeof(Scalar) / (1 << 20)) << " MiB, W "
            << (static_cast<double>(n) * nev * sizeof(Scalar) / (1 << 20)) << " MiB\n\n";

  // ---------------------------------------------------------------------
  // Host reference, kept for both the correctness check and the benchmark.
  //
  // The panel is gathered into a dense host mirror once, in its own scope so
  // the staging buffer is released before anything is timed: the gather is
  // an artifact of the block-of-blocks layout, not part of the product.
  // ---------------------------------------------------------------------
  RowMajorMatrix<Scalar> Vm;
  RowMajorMatrix<Scalar> Mm(ncv, nev);
  {
    const auto Vh = backend.host_block(V.panel_view(0, in_blocks), trl::Access::Read);
    const auto Mh = backend.host_block(M, trl::Access::Read);
    Vm = gather_panel<Scalar, blocksize>(Vh.data(), n, in_blocks);
    Mm = Eigen::Map<const RowMajorMatrix<Scalar>>(Mh.data(), ncv, nev);
  }

  const RowMajorMatrix<Scalar> ref = Vm * Mm;

  // ---------------------------------------------------------------------
  // Correctness check: both SYCL mult variants vs. the Eigen reference.
  // ---------------------------------------------------------------------
  auto check_against_ref = [&](const char* mode) {
    const auto Wh = backend.host_block(W.panel_view(0, out_blocks), trl::Access::Read);
    const auto got = gather_panel<Scalar, blocksize>(Wh.data(), n, out_blocks);

    // Relative to the column norm rather than entry-wise: entries of a random
    // GEMM pass through zero, and a relative test on those is meaningless.
    Scalar worst = 0;
    for (Eigen::Index j = 0; j < got.cols(); ++j) {
      const Scalar scale = ref.col(j).norm();
      const Scalar err = (got.col(j) - ref.col(j)).norm() / (scale > 0 ? scale : Scalar{1});
      worst = std::max(worst, err);
    }

    // Same tolerance as dot.cc. The device reduces in a different order than
    // Eigen, so expect ~sqrt(n) * eps of genuine disagreement; a real bug is
    // orders of magnitude larger than that.
    std::cout << "  " << mode << ": worst relative column error: " << worst << "\n";
    const bool correct = worst < 1e-10;
    std::cout << "  Result is " << (correct ? "correct" : "wrong") << "\n";
    return correct;
  };

  std::cout << "Correctness check:\n";

  V.panel_view(0, in_blocks).mult(trl::TransposeMode::NoTranspose, M, W.panel_view(0, out_blocks));
  q.wait();
  bool correct = check_against_ref("NoTranspose");

  // For TransposeMode::Transpose the kernel expects the transpose factor
  // stored row-major as (out.cols() x cols()), so we hand it M^T: the
  // mathematical product V * M -- and hence the reference -- is the same as
  // above, which is exactly what makes the two kernel paths comparable.
  RowMajorMatrix<Scalar> Mtm = Mm.transpose();
  OwnedDenseMatrix<Scalar> Mt(q, nev, ncv);
  {
    auto host = backend.host_block(Mt, trl::Access::Write);
    std::copy_n(Mtm.data(), static_cast<std::size_t>(Mtm.size()), host.data());
  }

  V.panel_view(0, in_blocks).mult(trl::TransposeMode::Transpose, Mt, W.panel_view(0, out_blocks));
  q.wait();
  correct = check_against_ref("Transpose") && correct;

  std::cout << "\n";
  if (!correct) return EXIT_FAILURE;

  // ---------------------------------------------------------------------
  // Benchmark: Eigen on the host.
  // ---------------------------------------------------------------------
  volatile Scalar sink = 0; // accumulate results so the compiler cannot elide the work
  RowMajorMatrix<Scalar> res(n, nev);   // preallocated, so only the product is timed
  Result eigen_result = time_wall("Eigen mult (host wall time)", warmups, repeats, [&] {
    res.noalias() = Vm * Mm;
    sink += res(0, 0);
  });
  (void)sink;

  // ---------------------------------------------------------------------
  // Benchmark: SYCL kernels, timed with host wall time and SYCL event profiling.
  // ---------------------------------------------------------------------
  std::vector<sycl::event> events;
  std::vector<double> kernel_times_ms;
  kernel_times_ms.reserve(static_cast<std::size_t>(repeats));

  Result sycl_wall_result = time_wall("SYCL mult (host wall time)", warmups, repeats, [&] {
    events.clear();
    V.panel_view(0, in_blocks).mult(trl::TransposeMode::NoTranspose, M, W.panel_view(0, out_blocks), &events);
    q.wait();
    kernel_times_ms.push_back(total_kernel_ms(events));
  });

  Result sycl_kernel_result = summarize("SYCL mult (event profiling)", kernel_times_ms);

  // ---------------------------------------------------------------------
  // Report.
  // ---------------------------------------------------------------------
  std::cout << "Benchmark results (" << repeats << " repeats, " << warmups << " warmups):\n";
  report(eigen_result);
  report(sycl_wall_result);
  report(sycl_kernel_result);

  const double eigen_min = eigen_result.min_ms;
  const double sycl_wall_min = sycl_wall_result.min_ms;
  const double sycl_kernel_min = sycl_kernel_result.min_ms;
  std::cout << "\nMin-based comparison:\n";
  std::cout << "  Eigen / SYCL (wall time):        " << (sycl_wall_min > 0 ? eigen_min / sycl_wall_min : 0.0) << "x\n";
  std::cout << "  Eigen / SYCL (kernel time only): " << (sycl_kernel_min > 0 ? eigen_min / sycl_kernel_min : 0.0) << "x\n";

  // ---------------------------------------------------------------------
  // Compute percentage of roofline max
  // ---------------------------------------------------------------------
  // V is read once and W written once; M is (ncv x nev) doubles, negligible
  // next to n and assumed cache-resident, so it is not counted.
  const double flops = 2.0 * static_cast<double>(n) * ncv * nev;
  const double bytes = static_cast<double>(sizeof(Scalar)) * static_cast<double>(n) * (ncv + nev);
  const double ai = flops / bytes;
  std::cout << "Arithmetic intensity: " << ai << "\n";

  // Measure achievable peaks on this device/queue instead of hardcoding them.
  const auto peaks = trl::benchmark::measure_peaks<Scalar>(q);
  std::cout << "Measured peaks:\n";
  std::cout << "  STREAM triad: " << peaks.triad_gbps << " GB/s\n";
  std::cout << "  Copy stream:  " << peaks.copy_gbps << " GB/s\n";
  std::cout << "  Read stream:  " << peaks.read_gbps << " GB/s\n";
  std::cout << "  FMA:          " << peaks.fma_gflops << " GFLOP/s\n";

  // mult streams one array in and one out, so the copy probe is the matching
  // bandwidth yardstick (the read probe would flatter it, the triad penalise it).
  const double peak_bw = peaks.copy_gbps * 1e9;    // Bytes/s
  const double peak_flops = peaks.fma_gflops * 1e9; // FLOP/s

  const double roofline = std::min(peak_flops, peak_bw * ai);
  const double kernel = flops / (sycl_kernel_min * 1e-3);

  // Unlike dot (AI = 0.5), this shape can sit on either side of the ridge:
  // at ncv = 64, nev = 16 the AI is 3.2, which is compute bound on a GPU with
  // 1/64 FP64 and bandwidth bound on a CPU.
  std::cout << "  Ridge point:  " << peak_flops / peak_bw << " FLOP/byte (" << (ai > peak_flops / peak_bw ? "compute" : "bandwidth") << " bound here)\n";
  std::cout << "  Roofline: " << roofline << " FLOP/s\n";
  std::cout << "  Kernel:   " << kernel << " FLOP/s\n";
  std::cout << "  K / R:    " << kernel / roofline << "\n";
  std::cout << "  Kernel bandwidth: " << (bytes / (sycl_kernel_min * 1e-3)) / 1e9 << " GB/s ("
            << 100.0 * (bytes / (sycl_kernel_min * 1e-3)) / peak_bw << "% of copy stream)\n";
}
