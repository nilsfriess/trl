#include <cmath>
#include <iostream>
#include <numeric>

#include <trl/concepts.hh>
#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>

#include "../test_lanczos_convergence.hh"
#include "ginkgo/diagonal.hh"
#include "ginkgo/laplace.hh"
#include "ginkgo_test_helper.hh"

template <class Scalar, unsigned int bs>
bool run_test_diagonal()
{
  std::cout << "\n[Diagonal Test: bs=" << bs << "]\n";
  trl::ScopedTimer timer;
  auto exec = gko::OmpExecutor::create();

  const unsigned int N = 256;
  using EVP = DiagonalEigenproblem<Scalar, bs>;
  GinkgoTestHelper<EVP> helper;
  auto evp = std::make_shared<EVP>(exec, N);

  std::vector<Scalar> exact_eigenvalues(16);
  std::iota(exact_eigenvalues.begin(), exact_eigenvalues.end(), 1);

  return test_lanczos_convergence(evp, helper, exact_eigenvalues);
}

template <class Scalar, unsigned int bs>
bool run_test_laplace()
{
  std::cout << "\n[Laplace Test: bs=" << bs << "]\n";
  trl::ScopedTimer timer;
  auto exec = gko::OmpExecutor::create();

  const unsigned int N = 256;
  using EVP = Laplace1DEigenproblem<Scalar, bs>;
  GinkgoTestHelper<EVP> helper;
  auto evp = std::make_shared<EVP>(exec, N);

  std::vector<Scalar> exact_eigenvalues(16);
  for (size_t i = 0; i < exact_eigenvalues.size(); ++i) exact_eigenvalues[i] = 2.0 - 2.0 * std::cos((i + 1) * M_PI / (N + 1));

  return test_lanczos_convergence(evp, helper, exact_eigenvalues);
}

int main()
{
  std::cout << "========================================\n";
  std::cout << "    Lanczos Convergence Tests          \n";
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
