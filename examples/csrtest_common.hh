#pragma once

#include <Eigen/Core>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>

template <unsigned int BlockSize, class CreateEvp, class PrintBackendInfo, class InitializeBlock, class Synchronize>
int run_csrtest(int argc, char* argv[], CreateEvp&& create_evp, PrintBackendInfo&& print_backend_info, InitializeBlock&& initialize_block,
                Synchronize&& synchronize)
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix.mtx> [nev] [ncv]\n";
    std::cerr << "  matrix.mtx: Path to a Matrix Market file\n";
    std::cerr << "  nev: Number of eigenvalues to compute (default: 10)\n";
    std::cerr << "  ncv: Number of Lanczos vectors (default: 4 * nev)\n";
    return 1;
  }

  const std::string matrix_file = argv[1];
  const unsigned int nev = (argc > 2) ? std::atoi(argv[2]) : 16;
  const unsigned int ncv = (argc > 3) ? std::atoi(argv[3]) : 4 * nev;

  std::cout << "========================================\n";
  std::cout << "    CSR Sparse Matrix Eigensolver      \n";
  std::cout << "========================================\n";
  std::cout << "Matrix file: " << matrix_file << "\n";
  std::cout << "nev = " << nev << ", ncv = " << ncv << ", blocksize = " << BlockSize << "\n";

  try {
    auto evp = create_evp(matrix_file);
    print_backend_info();

    using EVP = std::decay_t<decltype(*evp)>;
    using Scalar = typename EVP::Scalar;

    std::cout << "Matrix size: " << evp->size() << " x " << evp->size() << "\n\n";

    trl::EigensolverParams params{.nev = nev, .ncv = ncv, .max_restarts = 1000, .tolerance = 1e-5};

    Eigen::SparseMatrix<Scalar> A;
    Eigen::loadMarket(A, matrix_file);

    using Op = Spectra::SparseSymMatProd<Scalar>;
    Op op(A);
    constexpr std::size_t num_runs = 1;

    Eigen::VectorX<Scalar> spectra_eigenvalues;
    std::chrono::duration<double, std::milli> spectra_total{0};
    for (std::size_t i = 0; i < num_runs; ++i) {
      Spectra::SymEigsSolver<Op> eigs(op, params.nev, params.ncv);
      eigs.init();

      std::cout << "Running Spectra...\n";
      const auto start = std::chrono::steady_clock::now();
      eigs.compute();
      const auto end = std::chrono::steady_clock::now();
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      spectra_total += elapsed;
      std::cout << "Run " << i << ", took: " << elapsed.count() << "ms (" << eigs.num_iterations() << " iterations, " << eigs.num_operations()
                << " operator applications)\n";

      if (i + 1 == num_runs) {
        if (eigs.info() != Spectra::CompInfo::Successful) throw std::runtime_error("Spectra failed to converge");
        spectra_eigenvalues = eigs.eigenvalues();
      }
    }

    const double spectra_avg = spectra_total.count() / static_cast<double>(num_runs);
    std::cout << "Spectra average over " << num_runs << " runs: " << spectra_avg << "ms\n";
    std::cout << "Eigenvalues found using spectra:\n" << std::fixed << std::setprecision(10) << spectra_eigenvalues << std::endl;

    std::chrono::duration<double, std::milli> lanczos_total{0};
    trl::EigensolverResult last_result{};
    for (std::size_t i = 0; i < num_runs; ++i) {
      trl::BlockLanczos<EVP, trl::TwiceModifiedGS> lanczos(evp, params);

      auto V0 = lanczos.initial_block();
      std::mt19937 rng(42);
      std::normal_distribution<double> dist;
      initialize_block(V0, rng, dist);

      std::cout << "Running Block Lanczos...\n";
      const auto start = std::chrono::steady_clock::now();
      last_result = lanczos.solve();
      synchronize();
      const auto end = std::chrono::steady_clock::now();
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      lanczos_total += elapsed;
      std::cout << "Run " << i << ", took: " << elapsed.count() << "ms\n";
    }

    const double lanczos_avg = lanczos_total.count() / static_cast<double>(num_runs);
    std::cout << "Block Lanczos average over " << num_runs << " runs: " << lanczos_avg << "ms\n";

    if (last_result.converged) {
      std::cout << "\nConverged in " << last_result.iterations << " iterations";
      std::cout << " (" << last_result.n_op_apply << " matrix-vector products)\n\n";
    }
    else {
      std::cout << "\nDid not fully converge after " << last_result.iterations << " iterations\n\n";
    }

    synchronize();
    const auto computed_eigenvalues = evp->get_current_eigenvalues();

    std::cout << "Computed eigenvalues (largest " << nev << "):\n";
    std::cout << std::fixed << std::setprecision(10);
    for (unsigned int i = 0; i < nev; ++i) std::cout << "  λ[" << std::setw(3) << i << "] = " << computed_eigenvalues[i] << "\n";

    std::cout << "Difference between ours and Spectra:\n";
    for (std::size_t i = 0; i < params.nev; ++i) {
      const double reference = std::abs(spectra_eigenvalues[i]);
      const double error = std::abs(computed_eigenvalues[i] - spectra_eigenvalues[i]);
      std::cout << "  Eigenvalue " << i << ": " << (reference > 0.0 ? error / reference : error) << "\n";
    }

    std::cout << "\n========================================\n";
  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
