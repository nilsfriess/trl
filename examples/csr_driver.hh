#pragma once

// Shared driver for the csrtest examples: load a Matrix Market file, compute a
// reference solution with Spectra, run Block Lanczos against it, and compare.
// Everything here is backend-generic; the backend and operator are supplied by
// the caller.

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include <trl/concepts.hh>
#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>

#include <Eigen/Core>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>

namespace trl::examples {

struct CsrTestOptions {
  std::string matrix_file;
  unsigned int nev = 16;
  unsigned int ncv = 0; // 0 -> 4 * nev
  std::size_t num_runs = 1;
};

/** @brief Parses `<matrix.mtx> [nev] [ncv] [num_runs]`.
 *
 *  @returns false if the arguments are unusable, after printing usage.
 */
inline bool parse_args(int argc, char* argv[], CsrTestOptions& opts)
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix.mtx> [nev] [ncv] [num_runs]\n";
    std::cerr << "  matrix.mtx: Path to a Matrix Market file\n";
    std::cerr << "  nev: Number of eigenvalues to compute (default: 16)\n";
    std::cerr << "  ncv: Number of Lanczos vectors (default: 4 * nev)\n";
    std::cerr << "  num_runs: Number of timed repetitions (default: 1)\n";
    return false;
  }

  opts.matrix_file = argv[1];
  if (argc > 2) opts.nev = std::atoi(argv[2]);
  if (argc > 3) opts.ncv = std::atoi(argv[3]);
  if (argc > 4) opts.num_runs = std::atoi(argv[4]);
  if (opts.ncv == 0) opts.ncv = 4 * opts.nev;

  return true;
}

/** @brief Runs the comparison. Returns a process exit code. */
template <trl::Backend B, trl::Operator<B> O>
int run_csr_test(B& backend, std::shared_ptr<O> op, const CsrTestOptions& opts)
{
  using Scalar = typename B::Scalar;
  constexpr auto bs = B::blocksize;

  std::cout << "========================================\n";
  std::cout << "    CSR Sparse Matrix Eigensolver      \n";
  std::cout << "========================================\n";
  std::cout << "Matrix file: " << opts.matrix_file << "\n";
  std::cout << "nev = " << opts.nev << ", ncv = " << opts.ncv << ", blocksize = " << bs << "\n";
  std::cout << "Matrix size: " << op->size() << " x " << op->size() << "\n\n";

  trl::EigensolverParams params{.nev = opts.nev, .ncv = opts.ncv, .max_restarts = 1000, .tolerance = 1e-8};

  // Reference solution from Spectra.
  Eigen::SparseMatrix<double> A;
  Eigen::loadMarket(A, opts.matrix_file);
  using SpectraOp = Spectra::SparseSymMatProd<double>;
  SpectraOp spectra_op(A);

  Eigen::VectorXd evalues;
  std::chrono::duration<double, std::milli> spectra_total{0};
  for (std::size_t i = 0; i < opts.num_runs; ++i) {
    Spectra::SymEigsSolver<SpectraOp> eigs(spectra_op, params.nev, params.ncv);
    eigs.init();

    std::cout << "Running Spectra...\n";
    auto start = std::chrono::steady_clock::now();
    eigs.compute();
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    spectra_total += elapsed;
    std::cout << "Run " << i << ", took: " << elapsed.count() << "ms\n";

    if (i + 1 == opts.num_runs && eigs.info() == Spectra::CompInfo::Successful) evalues = eigs.eigenvalues();
  }
  std::cout << "Spectra average over " << opts.num_runs << " runs: " << spectra_total.count() / static_cast<double>(opts.num_runs) << "ms\n";
  std::cout << "Eigenvalues found using spectra:\n" << std::fixed << std::setprecision(10) << evalues << std::endl;

  // Block Lanczos.
  std::chrono::duration<double, std::milli> lanczos_total{0};
  trl::EigensolverResult<Scalar> result{};

  for (std::size_t i = 0; i < opts.num_runs; ++i) {
    trl::BlockLanczos lanczos(backend, op, params);

    {
      auto V0 = lanczos.initial_block();
      auto host = backend.host_block(V0, Access::Write);

      std::mt19937 rng(42);
      std::normal_distribution<Scalar> dist;
      std::generate_n(host.data(), host.size(), [&]() { return dist(rng); });
    }

    std::cout << "Running Block Lanczos...\n";
    auto start = std::chrono::steady_clock::now();
    result = lanczos.solve();
    backend.sync();
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    lanczos_total += elapsed;
    std::cout << "Run " << i << ", took: " << elapsed.count() << "ms\n";
  }
  std::cout << "Block Lanczos average over " << opts.num_runs << " runs: " << lanczos_total.count() / static_cast<double>(opts.num_runs) << "ms\n";

  if (result.converged) std::cout << "\nConverged in " << result.iterations << " iterations (" << result.n_op_apply << " matrix-vector products)\n\n";
  else std::cout << "\nDid not fully converge after " << result.iterations << " iterations\n\n";

  const auto& eigenvalues = result.eigenvalues;
  if (eigenvalues.size() < params.nev) {
    std::cerr << "Only " << eigenvalues.size() << " eigenvalues available, expected " << params.nev << "\n";
    return 1;
  }

  std::cout << "Computed eigenvalues (largest " << params.nev << "):\n";
  std::cout << std::fixed << std::setprecision(10);
  for (unsigned int i = 0; i < params.nev; ++i) std::cout << "  λ[" << std::setw(3) << i << "] = " << eigenvalues[i] << "\n";

  // Relative error, matching the solver's own convergence criterion.
  std::cout << "Relative difference between ours and Spectra:\n";
  for (unsigned int i = 0; i < params.nev; ++i) std::cout << "  Eigenvalue " << i << ": " << std::abs(eigenvalues[i] - evalues[i]) / std::abs(evalues[i]) << "\n";

  std::cout << "\n========================================\n";
  return 0;
}

} // namespace trl::examples
