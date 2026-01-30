// Example: Compute largest eigenvalues of a sparse matrix from a Matrix Market file
//
// Usage: ./csrtest <matrix.mtx> [nev] [ncv]
//   matrix.mtx: Path to a Matrix Market file containing a symmetric sparse matrix
//   nev: Number of eigenvalues to compute (default: 10)
//   ncv: Number of Lanczos vectors (default: 4 * nev)

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <sycl/sycl.hpp>

#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>

#include "csrevp.hh"

constexpr unsigned int BLOCKSIZE = 8;

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
  unsigned int nev = (argc > 2) ? std::atoi(argv[2]) : 10;
  unsigned int ncv = (argc > 3) ? std::atoi(argv[3]) : 4 * nev;

  std::cout << "========================================\n";
  std::cout << "    CSR Sparse Matrix Eigensolver      \n";
  std::cout << "========================================\n";
  std::cout << "Matrix file: " << matrix_file << "\n";
  std::cout << "nev = " << nev << ", ncv = " << ncv << ", blocksize = " << BLOCKSIZE << "\n";

  try {
    sycl::queue queue{sycl::property::queue::in_order{}};
    std::cout << "SYCL device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n\n";

    // Create the eigenvalue problem from the matrix file
    using EVP = CSREVP<double, BLOCKSIZE>;
    auto evp = std::make_shared<EVP>(queue, matrix_file);

    std::cout << "Matrix size: " << evp->size() << " x " << evp->size() << "\n\n";

    // Set up the Lanczos solver
    trl::EigensolverParams params{
        .nev = nev,
        .ncv = ncv,
        .max_restarts = 1000,
    };
    trl::BlockLanczos lanczos(evp, params);

    // Initialize with random starting block
    auto V0 = lanczos.initial_block();
    std::mt19937 rng(42);
    std::normal_distribution<double> dist;
    std::generate_n(V0.data, V0.rows() * V0.cols(), [&]() { return dist(rng); });

    // Solve
    std::cout << "Running Block Lanczos...\n";
    auto result = lanczos.solve();
    queue.wait();

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
    for (unsigned int i = 0; i < nev; ++i) {
      std::cout << "  λ[" << std::setw(3) << i << "] = " << eigenvalues[i] << "\n";
    }

    std::cout << "\n========================================\n";
  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
