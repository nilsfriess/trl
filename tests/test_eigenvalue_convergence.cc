#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>

#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>

#include "../examples/diagonal.hh"
#include "../examples/laplace.hh"

// Test convergence of eigenvalues for DiagonalEVP
// Analytical eigenvalues are simply the diagonal entries: λ_i = i+1
template <unsigned int bs>
bool test_diagonal_convergence(sycl::queue& q, bool verbose)
{
  using Scalar = double;
  const int N = 64;
  const unsigned int nev = 8;
  const unsigned int ncv = 24;
  const Scalar tol = 1e-6;

  std::cout << "Testing DiagonalEVP convergence (bs=" << bs << ", N=" << N << "): ";

  auto evp = std::make_shared<DiagonalEVP<Scalar, bs>>(q, N);
  trl::EigensolverParams params{nev, ncv};
  trl::BlockLanczos<DiagonalEVP<Scalar, bs>> solver(evp, params);

  // Initialize with random starting vector
  auto V0 = solver.initial_block();
  std::mt19937 rng(42);
  std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < V0.rows() * V0.cols(); ++i) V0.data[i] = dist(rng);

  // Solve
  auto result = solver.solve();

  if (!result.converged) {
    std::cout << "Failed (not converged)\n";
    return false;
  }

  // Check eigenvalues against analytical values
  const auto* eigenvalues = solver.get_eigenvalues();
  std::vector<Scalar> exact_eigenvalues(N);
  for (int i = 0; i < N; ++i) exact_eigenvalues[i] = static_cast<Scalar>(i + 1);

  // Sort computed eigenvalues
  std::vector<Scalar> computed(eigenvalues, eigenvalues + nev);
  std::sort(computed.begin(), computed.end());

  bool passed = true;
  Scalar max_error = 0;

  if (verbose) {
    std::cout << "\n  Eigenvalue comparison:\n";
  }

  for (unsigned int i = 0; i < nev; ++i) {
    Scalar error = std::abs(computed[i] - exact_eigenvalues[i]);
    max_error = std::max(max_error, error);

    if (verbose) {
      std::cout << "    λ[" << i << "] = " << computed[i] << " (exact: " << exact_eigenvalues[i] << ", error: " << error << ")\n";
    }

    if (error > tol) {
      passed = false;
    }
  }

  if (passed) {
    std::cout << "Passed (max error: " << max_error << ")\n";
  } else {
    std::cout << "Failed (max error: " << max_error << " > " << tol << ")\n";
  }

  return passed;
}

// Test convergence of eigenvalues for 1D Laplacian
// Analytical eigenvalues: λ_k = 2(1 - cos(kπ/(N+1)))
template <unsigned int bs>
bool test_laplace_convergence(sycl::queue& q, bool verbose)
{
  using Scalar = double;
  const int N = 100;
  const unsigned int nev = 8;
  const unsigned int ncv = 32;
  const Scalar tol = 1e-6;

  std::cout << "Testing OneDLaplaceEVP convergence (bs=" << bs << ", N=" << N << "): ";

  auto evp = std::make_shared<trl::OneDLaplaceEVP<Scalar, bs>>(q, N);
  trl::EigensolverParams params{nev, ncv};
  trl::BlockLanczos<trl::OneDLaplaceEVP<Scalar, bs>> solver(evp, params);

  // Initialize with random starting vector
  auto V0 = solver.initial_block();
  std::mt19937 rng(42);
  std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < V0.rows() * V0.cols(); ++i) V0.data[i] = dist(rng);

  // Solve
  auto result = solver.solve();

  if (!result.converged) {
    std::cout << "Failed (not converged after " << result.iterations << " iterations)\n";
    return false;
  }

  // Compute analytical eigenvalues
  std::vector<Scalar> exact_eigenvalues(N);
  for (int i = 0; i < N; ++i) {
    exact_eigenvalues[i] = 2.0 * (1.0 - std::cos((i + 1) * M_PI / (N + 1)));
  }

  // Sort computed eigenvalues
  const auto* eigenvalues = solver.get_eigenvalues();
  std::vector<Scalar> computed(eigenvalues, eigenvalues + nev);
  std::sort(computed.begin(), computed.end());

  bool passed = true;
  Scalar max_error = 0;

  if (verbose) {
    std::cout << "\n  Eigenvalue comparison:\n";
  }

  for (unsigned int i = 0; i < nev; ++i) {
    Scalar error = std::abs(computed[i] - exact_eigenvalues[i]);
    max_error = std::max(max_error, error);

    if (verbose) {
      std::cout << "    λ[" << i << "] = " << computed[i] << " (exact: " << exact_eigenvalues[i] << ", error: " << error << ")\n";
    }

    if (error > tol) {
      passed = false;
    }
  }

  if (passed) {
    std::cout << "Passed (max error: " << max_error << ")\n";
  } else {
    std::cout << "Failed (max error: " << max_error << " > " << tol << ")\n";
  }

  return passed;
}

int main()
{
  sycl::queue q;
  bool verbose = false;
  int num_failed = 0;

  std::cout << "========================================\n";
  std::cout << "Eigenvalue Convergence Tests\n";
  std::cout << "========================================\n\n";

  // Test DiagonalEVP with different block sizes
  std::cout << "DiagonalEVP tests:\n";
  if (!test_diagonal_convergence<1>(q, verbose)) num_failed++;
  if (!test_diagonal_convergence<2>(q, verbose)) num_failed++;
  if (!test_diagonal_convergence<4>(q, verbose)) num_failed++;

  std::cout << "\n";

  // Test OneDLaplaceEVP with different block sizes
  std::cout << "OneDLaplaceEVP tests:\n";
  if (!test_laplace_convergence<1>(q, verbose)) num_failed++;
  if (!test_laplace_convergence<2>(q, verbose)) num_failed++;
  if (!test_laplace_convergence<4>(q, verbose)) num_failed++;

  std::cout << "\n========================================\n";
  if (num_failed == 0) {
    std::cout << "All convergence tests passed!\n";
    return 0;
  } else {
    std::cout << num_failed << " test(s) failed!\n";
    return 1;
  }
}
