#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sycl/sycl.hpp>

#include "laplace.hh"
#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>

int main()
{
  sycl::queue q;

  // Problem size
  const int N = 100;

  // Number of eigenvalues to compute
  const unsigned int nev = 8;

  // Size of Krylov subspace (must be larger than nev)
  const unsigned int ncv = 32;

  std::cout << "================================================\n";
  std::cout << "Block Lanczos Eigenvalue Solver Example\n";
  std::cout << "================================================\n";
  std::cout << "Problem: 1D Laplacian with Dirichlet boundary conditions\n";
  std::cout << "Size: " << N << " x " << N << "\n";
  std::cout << "Number of eigenvalues requested: " << nev << "\n";
  std::cout << "Krylov subspace size: " << ncv << "\n";
  std::cout << "================================================\n\n";

  // Create the 1D Laplacian eigenvalue problem
  auto evp = std::make_shared<trl::OneDLaplaceEVP<double, 1>>(q, N);

  // Set up eigensolver parameters
  trl::EigensolverParams params{nev, ncv};

  // Create the Block Lanczos solver
  trl::BlockLanczos<trl::OneDLaplaceEVP<double, 1>> solver(evp, params);

  // Initialize the starting vector with random values
  auto V0 = solver.initial_block();
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < V0.rows() * V0.cols(); ++i) V0.data[i] = dist(rng);

  // Solve the eigenvalue problem
  std::cout << "Starting Block Lanczos iteration...\n\n";
  auto result = solver.solve();

  std::cout << "\n================================================\n";
  std::cout << "Results\n";
  std::cout << "================================================\n";
  std::cout << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
  std::cout << "Iterations: " << result.iterations << "\n";
  std::cout << "Operator applications: " << result.n_op_apply << "\n";
  std::cout << "Converged eigenvalues: " << solver.get_nconv() << " / " << nev << "\n\n";

  // Display the computed eigenvalues
  const auto* eigenvalues = solver.get_eigenvalues();
  std::cout << "Computed eigenvalues:\n";
  for (unsigned int i = 0; i < std::min(nev, solver.get_nconv()); ++i) {
    // Analytical eigenvalues for 1D Laplacian: λ_k = 2(1 - cos(kπ/(N+1)))
    double exact = 2.0 * (1.0 - std::cos((i + 1) * M_PI / (N + 1)));
    double computed = eigenvalues[i];
    double error = std::abs(computed - exact);

    std::cout << "  λ[" << i << "] = " << std::scientific << std::setprecision(10) << computed << "  (exact: " << exact << ", error: " << error << ")\n";
  }

  std::cout << "\n================================================\n";

  return 0;
}
