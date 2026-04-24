// Example: Compute largest eigenvalues of a graded tridiagonal matrix using a matrix-free
// matvec kernel.
//
// The matrix is symmetric tridiagonal with A[i,i] = N-i and A[i,i+/-1] = -1.
// Eigenvalues are approximately N, N-1, ..., 1, so consecutive top eigenvalues are separated
// by ~1 regardless of N — much better conditioned than the 1D Laplacian whose top
// eigenvalues cluster near 4 with gaps of O(1/N^2).
//
// Usage: ./laptest [N] [nev] [ncv]
//   N:   Matrix size (default: 1000)
//   nev: Number of eigenvalues to compute (default: 16)
//   ncv: Number of Lanczos vectors (default: 4 * nev)

#include <sycl/sycl.hpp>

#include "graded.hh"
#include "trl/impl/sycl/profiling.hh"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

constexpr unsigned int BLOCKSIZE = 4;

int main(int argc, char* argv[])
{
  const unsigned int N = (argc > 1) ? std::atoi(argv[1]) : 1000;
  const unsigned int nev = (argc > 2) ? std::atoi(argv[2]) : 16;
  const unsigned int ncv = (argc > 3) ? std::atoi(argv[3]) : 4 * nev;

  std::cout << "========================================\n";
  std::cout << "  Matrix-free Graded Tridiagonal Eigensolver \n";
  std::cout << "========================================\n";

  sycl::queue queue{{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
  std::cout << "SYCL device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "N = " << N << ", nev = " << nev << ", ncv = " << ncv << ", blocksize = " << BLOCKSIZE << "\n\n";

  // --- Build the explicit sparse tridiagonal matrix for Spectra (reference) ---
  // Same operator as GradedTridiagEVP: diag = N-i, off-diag = -1
  Eigen::SparseMatrix<double> A(N, N);
  {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(3 * N);
    for (unsigned int i = 0; i < N; ++i) {
      triplets.emplace_back(i, i, static_cast<double>(N - i));
      if (i > 0) triplets.emplace_back(i, i - 1, -1.0);
      if (i < N - 1) triplets.emplace_back(i, i + 1, -1.0);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
  }

  using Op = Spectra::SparseSymMatProd<double>;
  Op op(A);

  Spectra::SymEigsSolver<Op> eigs(op, nev, ncv);
  eigs.init();

  double tolerance = 1e-3;
  std::cout << "Running Spectra (reference)...\n";
  const auto spectra_start = std::chrono::steady_clock::now();
  eigs.compute(Spectra::SortRule::LargestMagn, 1000, tolerance);
  const auto spectra_end = std::chrono::steady_clock::now();
  const auto spectra_ms = std::chrono::duration_cast<std::chrono::milliseconds>(spectra_end - spectra_start).count();
  const auto spectra_its = eigs.num_iterations();
  const auto spectra_nop = eigs.num_operations();
  std::cout << "Spectra took: " << spectra_ms << "ms (" << spectra_its << " iterations, " << spectra_nop << " operations)\n";

  if (eigs.info() != Spectra::CompInfo::Successful) {
    std::cerr << "Spectra failed to converge\n";
    return 1;
  }

  const Eigen::VectorXd spectra_eigenvalues = eigs.eigenvalues();
  std::cout << "Eigenvalues found using Spectra:\n" << std::fixed << std::setprecision(10) << spectra_eigenvalues << "\n\n";

  // --- Run Block Lanczos with the matrix-free graded tridiagonal ---
  using EVP = GradedTridiagEVP<double, BLOCKSIZE>;
  auto evp = std::make_shared<EVP>(queue, N);

  trl::EigensolverParams params{.nev = nev, .ncv = ncv, .max_restarts = 1000, .tolerance = tolerance};
  trl::BlockLanczos<EVP, trl::TwiceModifiedGS> lanczos(evp, params);

  auto V0 = lanczos.initial_block();
  std::mt19937 rng(42);
  std::normal_distribution<double> dist;
  const std::size_t v0_size = V0.rows() * V0.cols();
  std::vector<double> v0_host(v0_size);
  std::generate_n(v0_host.begin(), v0_size, [&]() { return dist(rng); });
  queue.memcpy(V0.data, v0_host.data(), v0_size * sizeof(double)).wait();

  std::cout << "Running Block Lanczos (matrix-free)...\n";
  const auto lanczos_start = std::chrono::steady_clock::now();
  const auto result = lanczos.solve();
  queue.wait();
  const auto lanczos_end = std::chrono::steady_clock::now();
  const auto lanczos_ms = std::chrono::duration_cast<std::chrono::milliseconds>(lanczos_end - lanczos_start).count();
  std::cout << "Block Lanczos took: " << lanczos_ms << "ms\n";

  if (result.converged) std::cout << "\nConverged in " << result.iterations << " iterations (" << result.n_op_apply << " matrix-vector products)\n\n";
  else std::cout << "\nDid not fully converge after " << result.iterations << " iterations\n\n";

  queue.wait();
  const auto computed_eigenvalues = evp->get_current_eigenvalues();

  std::cout << "Computed eigenvalues (largest " << nev << "):\n" << std::fixed << std::setprecision(10);
  for (unsigned int i = 0; i < nev; ++i) std::cout << "  λ[" << std::setw(3) << i << "] = " << computed_eigenvalues[i] << "\n";

  std::cout << "\nDifference between ours and Spectra:\n";
  for (unsigned int i = 0; i < nev; ++i) {
    const double reference = std::abs(spectra_eigenvalues[i]);
    const double error = std::abs(computed_eigenvalues[i] - spectra_eigenvalues[i]);
    std::cout << "  Eigenvalue " << i << ": " << (reference > 0.0 ? error / reference : error) << "\n";
  }

  std::cout << "\n========================================\n";

  trl::sycl::SyclProfiler::get().report();
  return 0;
}
