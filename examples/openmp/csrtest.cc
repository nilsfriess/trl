// Example: Compute largest eigenvalues of a sparse matrix from a Matrix Market file
//
// Usage: ./csrtest <matrix.mtx> [nev] [ncv]
//   matrix.mtx: Path to a Matrix Market file containing a symmetric sparse matrix
//   nev: Number of eigenvalues to compute (default: 10)
//   ncv: Number of Lanczos vectors (default: 4 * nev)

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
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>

#include "csrevp.hh"

constexpr unsigned int BLOCKSIZE = 4;

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix.mtx> [nev] [ncv]\n";
    std::cerr << "  matrix.mtx: Path to a Matrix Market file\n";
    std::cerr << "  nev: Number of eigenvalues to compute (default: 10)\n";
    std::cerr << "  ncv: Number of Lanczos vectors (default: 4 * nev)\n";
    return 1;
  }

  std::string matrix_file = argv[1];
  unsigned int nev = (argc > 2) ? std::atoi(argv[2]) : 16;
  unsigned int ncv = (argc > 3) ? std::atoi(argv[3]) : 4 * nev;

  std::cout << "========================================\n";
  std::cout << "    CSR Sparse Matrix Eigensolver      \n";
  std::cout << "========================================\n";
  std::cout << "Matrix file: " << matrix_file << "\n";
  std::cout << "nev = " << nev << ", ncv = " << ncv << ", blocksize = " << BLOCKSIZE << "\n";

  try {
    // Create the eigenvalue problem from the matrix file
    using EVP = CSREVP<double, BLOCKSIZE>;
    auto evp = std::make_shared<EVP>(matrix_file);

    std::cout << "Matrix size: " << evp->size() << " x " << evp->size() << "\n\n";

    // Set up the Lanczos solver
    trl::EigensolverParams params{.nev = nev, .ncv = ncv, .max_restarts = 1000, .tolerance = 1e-8};

    // Create the Spectra solver object to compute the reference solution
    Eigen::SparseMatrix<double> A;
    Eigen::loadMarket(A, matrix_file);
    using Op = Spectra::SparseSymMatProd<double>;
    Op op(A);
    Spectra::SymEigsSolver<Op> eigs(op, params.nev, params.ncv);
    constexpr std::size_t num_runs = 1;

    Eigen::VectorXd evalues;
    std::chrono::duration<double, std::milli> spectra_total{0};
    for (std::size_t i = 0; i < num_runs; ++i) {
      Spectra::SymEigsSolver<Op> eigs(op, params.nev, params.ncv);
      eigs.init();

      std::cout << "Running Spectra...\n";
      auto start = std::chrono::steady_clock::now();
      eigs.compute();
      auto end = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      spectra_total += elapsed;
      std::cout << "Run " << i << ", took: " << elapsed.count() << "ms\n";

      if (i + 1 == num_runs && eigs.info() == Spectra::CompInfo::Successful) evalues = eigs.eigenvalues();
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
      std::generate_n(V0.data, V0.rows() * V0.cols(), [&]() { return dist(rng); });

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

    auto eigenvalues = evp->get_current_eigenvalues();
    for (unsigned int i = 0; i < nev; ++i) std::cout << "  λ[" << std::setw(3) << i << "] = " << eigenvalues[i] << "\n";

    // Compare with spectra results
    std::cout << "Difference between ours and Spectra:\n";
    for (std::size_t i = 0; i < params.nev; ++i) std::cout << "  Eigenvalue " << i << ": " << std::abs(eigenvalues[i] - evalues[i]) / std::abs(evalues[i]) << "\n";

    std::cout << "\n========================================\n";
  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
