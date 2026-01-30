#include <cmath>
#include <iostream>
#include <numeric>

#include <trl/concepts.hh>
#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>

#include "../test_lanczos_convergence.hh"
#include "sycl/diagonal.hh"
#include "sycl/laplace.hh"
#include "test_helper.hh"

template <class Scalar, unsigned int bs>
bool run_test_diagonal(sycl::queue q)
{
  std::cout << "\n[Diagonal Test: bs=" << bs << "]\n";
  trl::ScopedTimer timer;

  const unsigned int N = 128;
  using EVP = DiagonalEVP<Scalar, bs>;

  auto evp = std::make_shared<EVP>(q, N);

  SYCLTestHelper<EVP> helper(q);

  // Use nev that is a multiple of blocksize for proper testing
  const unsigned int nev = 16;
  std::vector<Scalar> exact_eigenvalues(nev);
  for (std::size_t i = 0; i < nev; ++i) exact_eigenvalues[i] = (N - i) * (N - i);

  return test_lanczos_convergence(evp, helper, exact_eigenvalues);
}

template <class Scalar, unsigned int bs>
bool run_test_laplace(sycl::queue q)
{
  std::cout << "\n[Laplace 1D Test: bs=" << bs << "]\n";
  trl::ScopedTimer timer;

  const unsigned int N = 128;
  using EVP = Laplace1DEVP<Scalar, bs>;

  auto evp = std::make_shared<EVP>(q, N);

  SYCLTestHelper<EVP> helper(q);

  // Use nev that is a multiple of blocksize for proper testing
  const unsigned int nev = 16;
  std::vector<Scalar> exact_eigenvalues(nev);
  // Exact eigenvalues: λ_k = 2 - 2*cos(k*π/(N+1)) for k = 1, 2, ..., N
  // We want the largest nev eigenvalues (descending order)
  for (std::size_t i = 0; i < nev; ++i)
    exact_eigenvalues[i] = 2.0 - 2.0 * std::cos((N - i) * M_PI / (N + 1));

  return test_lanczos_convergence(evp, helper, exact_eigenvalues);
}

int main()
{
  std::cout << "========================================\n";
  std::cout << "    Lanczos Convergence Tests          \n";
  std::cout << "========================================\n";

  int num_failed = 0;

  sycl::queue q{sycl::property::queue::in_order{}};

  std::cout << "\n--- Diagonal Eigenvalue Problem ---\n";
  // if (!run_test_diagonal<double, 1>(q)) num_failed++;
  // if (!run_test_diagonal<double, 2>(q)) num_failed++;
  if (!run_test_diagonal<double, 4>(q)) num_failed++;
  if (!run_test_diagonal<double, 8>(q)) num_failed++;

  std::cout << "\n--- Laplace 1D Eigenvalue Problem ---\n";
  if (!run_test_laplace<double, 4>(q)) num_failed++;
  if (!run_test_laplace<double, 8>(q)) num_failed++;

  std::cout << "\n========================================\n";
  std::cout << "Tests completed: " << (num_failed == 0 ? "ALL PASSED" : std::to_string(num_failed) + " FAILED") << "\n";
  std::cout << "========================================\n";

  return num_failed;
}
