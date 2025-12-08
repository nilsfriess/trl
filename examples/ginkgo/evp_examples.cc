#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "trl/concepts.hh"
#include "trl/eigensolvers/lanczos.hh"
#include "trl/eigensolvers/params.hh"
#include "trl/impl/gingko/multivector.hh"

#include <ginkgo/ginkgo.hpp>

#include "diagonal.hh"
#include "laplace.hh"

template <typename EVP>
int run_example(const std::string& problem_name, std::shared_ptr<EVP> evp, std::size_t nev, std::function<double(std::size_t, std::size_t)> exact_eigenvalue_fn)
{
  using Scalar = typename EVP::Scalar;
  constexpr auto bs = EVP::blocksize;

  std::cout << "\n========================================\n";
  std::cout << "Problem: " << problem_name << "\n";
  std::cout << "N = " << evp->size() << ", bs = " << bs << ", nev = " << nev << "\n";
  std::cout << "========================================\n";

  trl::EigensolverParams params{
      .nev = static_cast<unsigned int>(nev),
      .ncv = static_cast<unsigned int>(4 * nev),
  };

  trl::BlockLanczos lanczos(evp, params);

  // Initialize first block randomly
  {
    auto V0 = lanczos.get_basis().block_view(0);
    std::mt19937 gen(42);
    std::normal_distribution<Scalar> dist(0.0, 1.0);
    for (std::size_t i = 0; i < V0.rows(); ++i)
      for (std::size_t j = 0; j < V0.cols(); ++j) V0.data()->get_values()[i * V0.cols() + j] = dist(gen);
  }

  auto result = lanczos.solve();

  // Check convergence
  if (result.converged) { std::cout << "\n✓ Solver converged after " << result.iterations << " iterations\n"; }
  else {
    std::cout << "\n✗ Solver did NOT converge after " << result.iterations << " iterations\n";
    return 1;
  }

  // Print out the computed eigenvalues and errors
  const auto& eigvals = evp->get_current_eigenvalues();
  std::cout << "\nComputed eigenvalues and errors:\n";
  std::cout << "  Index    Computed        Exact          Error\n";
  std::cout << "  -----    --------        -----          -----\n";
  double max_error = 0.0;
  for (std::size_t i = 0; i < params.nev; ++i) {
    double computed = eigvals.get_const_data()[i];
    double exact = exact_eigenvalue_fn(i, evp->size());
    double error = std::abs(computed - exact);
    max_error = std::max(max_error, error);
    std::cout << "  " << std::setw(5) << i << "    " << std::setw(10) << std::scientific << std::setprecision(6) << computed << "    " << std::setw(10) << std::scientific << std::setprecision(6)
              << exact << "    " << std::setw(10) << std::scientific << std::setprecision(2) << error << "\n";
  }
  std::cout << "\nMaximum error: " << std::scientific << std::setprecision(2) << max_error << "\n";

  return 0;
}

void print_usage(const char* prog_name)
{
  std::cout << "Usage: " << prog_name << " [OPTIONS]\n\n";
  std::cout << "Options:\n";
  std::cout << "  -p, --problem TYPE   Problem type: 'diagonal' or 'laplace' (default: both)\n";
  std::cout << "  -n, --size N         Matrix size (default: 256)\n";
  std::cout << "  -e, --nev NEV        Number of eigenvalues to compute (default: 16)\n";
  std::cout << "  -h, --help           Show this help message\n";
  std::cout << "\nExamples:\n";
  std::cout << "  " << prog_name << " -p diagonal\n";
  std::cout << "  " << prog_name << " -p laplace -n 512 -e 32\n";
  std::cout << "  " << prog_name << " --problem both\n";
  std::cout << "\nNote: Block size (bs=4) is set at compile time.\n";
}

int main(int argc, char* argv[])
{
  using Scalar = double;
  constexpr unsigned int bs = 4;

  std::string problem = "both";
  std::size_t N = 256;
  std::size_t nev = 16;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    }
    else if ((arg == "-p" || arg == "--problem") && i + 1 < argc) {
      problem = argv[++i];
      if (problem != "diagonal" && problem != "laplace" && problem != "both") {
        std::cerr << "Error: Problem type must be 'diagonal', 'laplace', or 'both'\n";
        return 1;
      }
    }
    else if ((arg == "-n" || arg == "--size") && i + 1 < argc) {
      N = std::atoi(argv[++i]);
      if (N == 0) {
        std::cerr << "Error: Matrix size must be positive\n";
        return 1;
      }
    }
    else if ((arg == "-e" || arg == "--nev") && i + 1 < argc) {
      nev = std::atoi(argv[++i]);
      if (nev == 0) {
        std::cerr << "Error: Number of eigenvalues must be positive\n";
        return 1;
      }
    }
    else {
      std::cerr << "Error: Unknown option '" << arg << "'\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  auto exec = gko::OmpExecutor::create();
  int total_failures = 0;

  // Run diagonal problem
  if (problem == "diagonal" || problem == "both") {
    using DiagEVP = DiagonalEigenproblem<Scalar, bs>;
    auto diag_evp = std::make_shared<DiagEVP>(exec, N);

    auto diagonal_exact = [](std::size_t i, std::size_t) -> double { return static_cast<double>(i + 1); };

    total_failures += run_example("Diagonal Eigenvalue Problem", diag_evp, nev, diagonal_exact);
  }

  // Run Laplace problem
  if (problem == "laplace" || problem == "both") {
    using LaplaceEVP = Laplace1DEigenproblem<Scalar, bs>;
    auto laplace_evp = std::make_shared<LaplaceEVP>(exec, N);

    auto laplace_exact = [](std::size_t i, std::size_t N) -> double { return 2.0 - 2.0 * std::cos((i + 1) * M_PI / (N + 1)); };

    total_failures += run_example("1D Laplacian (Dirichlet BC)", laplace_evp, nev, laplace_exact);
  }

  if (total_failures == 0) {
    std::cout << "\n========================================\n";
    std::cout << "All examples completed successfully!\n";
    std::cout << "========================================\n";
  }

  return total_failures;
}
