#include <cmath>
#include <iostream>
#include <numeric>

#include <trl/concepts.hh>
#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>

#include "../test_lanczos_convergence.hh"
#include "diagonal.hh"
#include "laplace.hh"
#include "test_helper.hh"

using trl::openmp::tests::DiagonalEVP;
using trl::openmp::tests::Laplace1DEVP;

template <class Scalar, unsigned int bs>
bool run_test_diagonal()
{
  std::cout << "\n[Diagonal Test: bs=" << bs << "]\n";
  trl::ScopedTimer timer;

  const unsigned int N = 128;
  using EVP = DiagonalEVP<Scalar, bs>;

  auto evp = std::make_shared<EVP>(N);

  OpenMPTestHelper<EVP> helper;

  const unsigned int nev = 16;
  std::vector<Scalar> exact_eigenvalues(nev);
  for (std::size_t i = 0; i < nev; ++i) exact_eigenvalues[i] = (N - i) * (N - i);

  const Scalar tol = Scalar(1e-8);
  return test_lanczos_convergence(evp, helper, exact_eigenvalues, tol);
}

template <class Scalar, unsigned int bs>
bool run_test_laplace()
{
  std::cout << "\n[Laplace 1D Test: bs=" << bs << "]\n";
  trl::ScopedTimer timer;

  const unsigned int N = 128;
  using EVP = Laplace1DEVP<Scalar, bs>;

  auto evp = std::make_shared<EVP>(N);

  OpenMPTestHelper<EVP> helper;

  const unsigned int nev = 16;
  std::vector<Scalar> exact_eigenvalues(nev);
  const Scalar pi = std::acos(Scalar(-1));
  for (std::size_t i = 0; i < nev; ++i) exact_eigenvalues[i] = Scalar(2) - Scalar(2) * std::cos((N - i) * pi / (N + 1));

  const Scalar tol = Scalar(1e-8);
  return test_lanczos_convergence(evp, helper, exact_eigenvalues, tol);
}

int main()
{
  std::cout << "========================================\n";
  std::cout << "    Lanczos Convergence Tests (OpenMP)  \n";
  std::cout << "========================================\n";

  int num_failed = 0;

  std::cout << "\n--- Diagonal Eigenvalue Problem ---\n";
  if (!run_test_diagonal<double, 1>()) num_failed++;
  if (!run_test_diagonal<double, 2>()) num_failed++;
  if (!run_test_diagonal<double, 4>()) num_failed++;
  if (!run_test_diagonal<double, 8>()) num_failed++;

  std::cout << "\n--- Laplace 1D Eigenvalue Problem ---\n";
  if (!run_test_laplace<double, 1>()) num_failed++;
  if (!run_test_laplace<double, 2>()) num_failed++;
  if (!run_test_laplace<double, 4>()) num_failed++;
  if (!run_test_laplace<double, 8>()) num_failed++;

  std::cout << "\n========================================\n";
  std::cout << "Tests completed: " << (num_failed == 0 ? "ALL PASSED" : std::to_string(num_failed) + " FAILED") << "\n";
  std::cout << "========================================\n";

  return num_failed;
}
