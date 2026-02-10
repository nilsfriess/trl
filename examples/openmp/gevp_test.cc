// Example: Compute eigenvalues of Ax = \lambda Bx using a shift-invert spectral transformation
//
// Usage: ./gevp_test <matrixA.mtx> <matrixB.mtx> [nev] [ncv] [shift]
//   matrixA.mtx: Path to a Matrix Market file containing a symmetric sparse matrix (the A matrix)
//   matrixB.mtx: Path to a Matrix Market file containing a symmetric positive semi-definite sparse matrix (the B matrix)
//   nev: Number of eigenvalues to compute (default: 16)
//   ncv: Number of Lanczos vectors (default: 4 * nev)
//   shift: Shift used in the shift-invert transformation (default: 1e-3)

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>

#include <Eigen/Core>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>

#include "Spectra/MatOp/SymShiftInvert.h"
#include "csrevp.hh"

constexpr unsigned int BLOCKSIZE = 1;

int main(int argc, char* argv[])
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <matrixA.mtx> <matrixB.mtx> [nev] [ncv]\n";
    std::cerr << "  matrixA.mtx: Path to a Matrix Market file (the A matrix)\n";
    std::cerr << "  matrixB.mtx: Path to a Matrix Market file (the B matrix)\n";
    std::cerr << "  nev: Number of eigenvalues to compute (default: 10)\n";
    std::cerr << "  ncv: Number of Lanczos vectors (default: 4 * nev)\n";
    return 1;
  }

  std::string matrix_file_A = argv[1];
  std::string matrix_file_B = argv[2];
  unsigned int nev = (argc > 3) ? std::atoi(argv[3]) : 16;
  unsigned int ncv = (argc > 4) ? std::atoi(argv[4]) : 4 * nev;
  double shift = (argc > 5) ? std::atof(argv[5]) : 1e-3;

  std::cout << "========================================\n";
  std::cout << "   CSR Sparse Generalized Eigensolver   \n";
  std::cout << "========================================\n";
  std::cout << "Matrix file A: " << matrix_file_A << "\n";
  std::cout << "Matrix file B: " << matrix_file_B << "\n";
  std::cout << "nev = " << nev << ", ncv = " << ncv << ", shift = " << shift << ", blocksize = " << BLOCKSIZE << "\n";

  try {
    // Create the eigenvalue problem from the matrix files
    using EVP = CSRGeneralizedEVP<double, BLOCKSIZE>;
    auto evp = std::make_shared<EVP>(matrix_file_A, matrix_file_B, shift);

    std::cout << "Matrix size: " << evp->size() << " x " << evp->size() << "\n\n";

    // Set up the Lanczos solver
    trl::EigensolverParams params{.nev = nev, .ncv = ncv, .max_restarts = 1000, .tolerance = 1e-8};

    // Create the Spectra solver object to compute the reference solution
    Eigen::SparseMatrix<double, Eigen::ColMajor> A;
    Eigen::SparseMatrix<double, Eigen::ColMajor> B;
    Eigen::loadMarket(A, matrix_file_A);
    Eigen::loadMarket(B, matrix_file_B);

    using Op = Spectra::SymShiftInvert<double>;
    using BOp = Spectra::SparseSymMatProd<double>;
    Op op(A, B);
    BOp Bop(B);
    Spectra::SymGEigsShiftSolver<Op, BOp, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, nev, ncv, shift);
    constexpr std::size_t num_runs = 1;

    Eigen::VectorXd evalues;
    std::chrono::duration<double, std::milli> spectra_total{0};
    for (std::size_t i = 0; i < num_runs; ++i) {
      geigs.init();

      std::cout << "Running Spectra...\n";
      auto start = std::chrono::steady_clock::now();
      geigs.compute(Spectra::SortRule::LargestMagn, 1000, 1e-8, Spectra::SortRule::SmallestAlge);
      auto end = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      spectra_total += elapsed;
      std::cout << "Run " << i << ", took: " << elapsed.count() << "ms\n";

      if (i + 1 == num_runs && geigs.info() == Spectra::CompInfo::Successful) evalues = geigs.eigenvalues();
    }

    const double spectra_avg = spectra_total.count() / static_cast<double>(num_runs);
    std::cout << "Spectra average over " << num_runs << " runs: " << spectra_avg << "ms\n";

    std::cout << "Eigenvalues found using spectra:\n" << std::fixed << std::setprecision(10) << evalues << std::endl;

    std::chrono::duration<double, std::milli> lanczos_total{0};
    trl::EigensolverResult last_result{};
    for (std::size_t i = 0; i < num_runs; ++i) {
      trl::BlockLanczos lanczos(evp, params);

      // Initialize with random starting block
      auto V0 = lanczos.initial_block();
      std::mt19937 rng(42);
      std::normal_distribution<double> dist;
      std::generate_n(V0.data_, V0.rows() * V0.cols(), [&]() { return dist(rng); });

      // Solve
      std::cout << "Running Block Lanczos...\n";
      auto start = std::chrono::steady_clock::now();
      last_result = lanczos.solve();
      auto end = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      lanczos_total += elapsed;
      std::cout << "Run " << i << ", took: " << elapsed.count() << "ms\n";
    }

    const double lanczos_avg = lanczos_total.count() / static_cast<double>(num_runs);
    std::cout << "Block Lanczos average over " << num_runs << " runs: " << lanczos_avg << "ms\n";

    auto result = last_result;
    if (result.converged) {
      std::cout << "\nConverged in " << result.iterations << " iterations";
      std::cout << " (" << result.n_op_apply << " matrix-vector products)\n\n";
    }
    else {
      std::cout << "\nDid not fully converge after " << result.iterations << " iterations\n\n";
    }

    // Print computed eigenvalues
    std::cout << "Computed eigenvalues (largest " << nev << "):\n";
    std::cout << std::fixed << std::setprecision(10);

    auto trl_evals_raw = evp->get_current_eigenvalues();
    std::vector<double> trl_evals_transformed;
    trl_evals_transformed.reserve(nev);
    for (unsigned int i = 0; i < nev; ++i) {
      double mu = trl_evals_raw[i];
      double val = shift + 1.0 / mu;
      trl_evals_transformed.push_back(val);
      std::cout << "  λ[" << std::setw(3) << i << "] = " << val << " (μ = " << mu << ")\n";
    }

    // Sort both sets of eigenvalues to compare them (Spectra and TRL might use different sorting)
    std::vector<double> spectra_evals_vec(evalues.data(), evalues.data() + evalues.size());
    std::sort(spectra_evals_vec.begin(), spectra_evals_vec.end());
    std::sort(trl_evals_transformed.begin(), trl_evals_transformed.end());

    // Compare with spectra results
    std::cout << "Difference between ours and Spectra (sorted):\n";
    for (std::size_t i = 0; i < params.nev; ++i) {
      double val_ours = trl_evals_transformed[i];
      double val_spectra = spectra_evals_vec[i];
      std::cout << "  Eigenvalue " << i << ": " << std::abs(val_ours - val_spectra) / std::abs(val_spectra) << "\n";
    }

    std::cout << "\n========================================\n";
  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
