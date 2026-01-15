#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "trl/concepts.hh"
#include "trl/eigensolvers/lanczos.hh"
#include "trl/eigensolvers/params.hh"
#include "trl/impl/gingko/multivector.hh"

#include <ginkgo/ginkgo.hpp>

#include "generalized_fem.hh"

void print_usage(const char* prog_name)
{
  std::cout << "Usage: " << prog_name << " [OPTIONS]\n\n";
  std::cout << "Solves generalized eigenvalue problem A x = λ B x using shift-invert Lanczos.\n\n";
  std::cout << "A = tridiag(-1, 2, -1)         (stiffness matrix)\n";
  std::cout << "B = (1/6) * tridiag(1, 4, 1)   (consistent mass matrix)\n\n";
  std::cout << "Options:\n";
  std::cout << "  -n, --size N         Matrix size (default: 256)\n";
  std::cout << "  -e, --nev NEV        Number of eigenvalues to compute (default: 16)\n";
  std::cout << "  -s, --sigma SIGMA    Shift parameter for shift-invert (default: 0.0)\n";
  std::cout << "  -h, --help           Show this help message\n";
  std::cout << "\nExamples:\n";
  std::cout << "  " << prog_name << " -n 512 -e 32\n";
  std::cout << "  " << prog_name << " --sigma 0.5     # Find eigenvalues near 0.5\n";
  std::cout << "\nNote: Block size (bs=4) is set at compile time.\n";
}

int main(int argc, char* argv[])
{
  using Scalar = double;
  constexpr unsigned int bs = 4;

  std::size_t N = 256;
  std::size_t nev = 16;
  Scalar sigma = 0.0;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
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
    else if ((arg == "-s" || arg == "--sigma") && i + 1 < argc) {
      sigma = std::atof(argv[++i]);
    }
    else {
      std::cerr << "Error: Unknown option '" << arg << "'\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  auto exec = gko::OmpExecutor::create();

  std::cout << "\n========================================\n";
  std::cout << "Generalized FEM Eigenvalue Problem\n";
  std::cout << "A x = λ B x (shift-invert transformation)\n";
  std::cout << "========================================\n";
  std::cout << "N = " << N << ", bs = " << bs << ", nev = " << nev << "\n";
  std::cout << "Shift σ = " << sigma << "\n";
  std::cout << "========================================\n";

  // Create the generalized eigenvalue problem with shift-invert
  using GenEVP = GeneralizedFEMEigenproblem<Scalar, bs>;
  auto evp = std::make_shared<GenEVP>(exec, N, sigma);

  // Verify: print smallest exact eigenvalues and corresponding shift-invert eigenvalues
  std::cout << "\nFirst few exact eigenvalues of A x = λ B x:\n";
  std::cout << "  Mode    λ (original)    μ = 1/(λ-σ) (shift-inv)\n";
  for (std::size_t i = 0; i < std::min(nev, std::size_t(8)); ++i) {
    Scalar lambda = GenEVP::exact_eigenvalue(i, N);
    Scalar mu = 1.0 / (lambda - sigma);
    std::cout << "  " << i << "       " << std::setprecision(10) << lambda 
              << "    " << mu << "\n";
  }
  
  // Quick test: verify operator on first eigenvector
  std::cout << "\n--- Verifying operator on first eigenvector ---\n";
  {
    auto test_x = evp->create_multivector(N, bs);
    auto test_y = evp->create_multivector(N, bs);
    auto test_z = evp->create_multivector(N, bs);
    auto x_view = test_x.block_view(0);
    auto y_view = test_y.block_view(0);
    auto z_view = test_z.block_view(0);
    
    // Set x to first eigenvector: sin(j*π/(N+1)) for j=1,...,N
    for (std::size_t j = 0; j < N; ++j) {
      Scalar val = std::sin((j + 1) * M_PI / (N + 1));
      for (unsigned int col = 0; col < bs; ++col) {
        x_view.data()->at(j, col) = (col == 0) ? val : 0.0;
        z_view.data()->at(j, col) = 0.0;
      }
    }
    
    // Apply operator: y = (A - σB)^{-1} B x
    evp->apply(x_view, y_view);
    
    // Check if y is a scalar multiple of x (i.e., x is eigenvector)
    Scalar y0 = y_view.data()->at(0, 0);
    Scalar x0 = x_view.data()->at(0, 0);
    Scalar ratio = y0 / x0;
    Scalar expected_mu = 1.0 / GenEVP::exact_eigenvalue(0, N);
    
    std::cout << "  Computed μ = y/x = " << ratio << "\n";
    std::cout << "  Expected  μ = 1/λ_0 = " << expected_mu << "\n";
    std::cout << "  Relative error: " << std::abs(ratio - expected_mu) / expected_mu << "\n";
  }
  std::cout << "--- End verification ---\n";

  // Set up Lanczos parameters
  trl::EigensolverParams params{
      .nev = static_cast<unsigned int>(nev),
      .ncv = static_cast<unsigned int>(3 * nev),
      .max_restarts = 100,
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

  std::cout << "\nRunning shift-invert Lanczos...\n";
  auto result = lanczos.solve();

  // Check convergence
  if (result.converged) { std::cout << "\n✓ Solver converged after " << result.iterations << " iterations\n"; }
  else {
    std::cout << "\n✗ Solver did NOT converge after " << result.iterations << " iterations\n";
    return 1;
  }

  std::cout << "  Total operator applications: " << result.n_op_apply << "\n";

  // Get the shift-invert eigenvalues (μ)
  const auto& mu_values = evp->get_current_eigenvalues();

  // Build a list of all exact eigenvalues
  std::vector<Scalar> exact_eigenvalues;
  for (std::size_t k = 0; k < N; ++k) {
    exact_eigenvalues.push_back(GenEVP::exact_eigenvalue(k, N));
  }

  // Convert computed eigenvalues to original eigenvalues: λ = σ + 1/μ
  std::vector<Scalar> computed_eigenvalues;
  for (std::size_t i = 0; i < params.nev; ++i) {
    Scalar mu = mu_values.get_const_data()[i];
    computed_eigenvalues.push_back(evp->convert_eigenvalue(mu));
  }

  // For each computed eigenvalue, find the closest exact eigenvalue
  std::cout << "\nComputed eigenvalues and errors:\n";
  std::cout << "  Index    μ (shift-inv)   λ (original)    λ (exact)       Mode    Error\n";
  std::cout << "  -----    -------------   ------------    ---------       ----    -----\n";

  double max_error = 0.0;
  std::vector<bool> exact_used(N, false);  // Track which exact eigenvalues have been matched

  // The shift-invert transformation maps:
  //   λ = σ + 1/μ
  // Eigenvalues closest to σ have largest |μ|.
  // For σ = 0, smallest λ (eigenvalues near 0) have largest μ.
  //
  // Note: Lanczos finds extremal eigenvalues, so for shift-invert it finds
  // eigenvalues closest to the shift.

  for (std::size_t i = 0; i < params.nev; ++i) {
    Scalar mu = mu_values.get_const_data()[i];
    Scalar lambda_computed = computed_eigenvalues[i];

    // Find the closest unmatched exact eigenvalue
    std::size_t best_mode = 0;
    Scalar best_error = std::numeric_limits<Scalar>::max();
    for (std::size_t k = 0; k < N; ++k) {
      if (!exact_used[k]) {
        Scalar err = std::abs(lambda_computed - exact_eigenvalues[k]);
        if (err < best_error) {
          best_error = err;
          best_mode = k;
        }
      }
    }
    exact_used[best_mode] = true;
    
    Scalar lambda_exact = exact_eigenvalues[best_mode];
    Scalar error = best_error;
    max_error = std::max(max_error, error);

    std::cout << "  " << std::setw(5) << i << "    " << std::setw(13) << std::scientific << std::setprecision(6) << mu << "    " << std::setw(12) << std::scientific << std::setprecision(6)
              << lambda_computed << "    " << std::setw(12) << std::scientific << std::setprecision(6) << lambda_exact << "    " << std::setw(4) << best_mode << "    " << std::setw(10) << std::scientific << std::setprecision(2) << error
              << "\n";
  }

  std::cout << "\nMaximum error: " << std::scientific << std::setprecision(2) << max_error << "\n";

  // Determine success
  bool success = (max_error < 1e-6);

  if (success) {
    std::cout << "\n========================================\n";
    std::cout << "✓ Generalized EVP example completed successfully!\n";
    std::cout << "========================================\n";
    return 0;
  }
  else {
    std::cout << "\n========================================\n";
    std::cout << "✗ Large errors detected. Test failed.\n";
    std::cout << "========================================\n";
    return 1;
  }
}
