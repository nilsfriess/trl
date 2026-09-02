#pragma once

#include <trl/concepts.hh>
#include <trl/eigensolvers/lanczos.hh>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "helpers.hh"
#include "test_helper.hh"

template <trl::Backend B, trl::Operator<B> O>
bool test_lanczos_convergence(B& backend, std::shared_ptr<O> op, const std::vector<typename B::Scalar>& exact_eigenvalues, typename B::Scalar tolerance = 1e-8)
{
  using Scalar = typename B::Scalar;
  constexpr auto bs = B::blocksize;
  auto N = op->size();

  std::cout << "\nTesting Lanczos convergence, type = " << trl::type_str<Scalar>() << ", N = " << N << ", bs = " << bs << ": ";

  unsigned int nev = exact_eigenvalues.size();
  trl::EigensolverParams params{.nev = nev, .ncv = 4 * nev, .max_restarts = 1000, .tolerance = static_cast<double>(tolerance)};
  trl::BlockLanczos lanczos(backend, op, params);

  auto V0 = lanczos.initial_block();
  trl::test::set_random(backend, V0);

  bool passed = true;

  auto res = lanczos.solve();
  backend.sync();

  if (res.converged) {
    std::cout << "Eigensolver converged in " << res.iterations << " iterations (" << res.n_op_apply << " operator applications).\n";
    std::cout << "Checking computed eigenvalues against exact values...\n";

    const auto& ev = res.eigenvalues;
    if (ev.size() < exact_eigenvalues.size()) {
      passed = false;
      std::cout << "In test_lanczos_convergence: Number of computed eigenvalues is smaller than number of requested eigenvalues, " << ev.size() << " vs. " << exact_eigenvalues.size() << "\n";
    }
    for (unsigned int i = 0; i < nev; ++i) {
      if (std::abs(ev[i] - exact_eigenvalues[i]) > tolerance * std::abs(exact_eigenvalues[i])) {
        std::cout << "In test_lanczos_convergence: Eigenvalue " << i << " differs from exact, computed " << ev[i] << ", expected " << exact_eigenvalues[i] << ", error "
                  << std::abs(ev[i] - exact_eigenvalues[i]) << "\n";
        passed = false;
      }
      else {
        std::cout << "  Eigenvalue " << i << " correct: " << ev[i] << ", error " << std::abs(ev[i] - exact_eigenvalues[i]) << "\n";
      }
    }
  }
  else passed = false;

  std::cout << (passed ? " " : "not ") << "passed.\n";

  return passed;
}

/** @brief Diagonal operator: exact eigenvalues are (N - i)^2, largest first. */
template <class Fixture, class Scalar, unsigned int bs>
bool run_convergence_diagonal()
{
  std::cout << "\n[Diagonal Test: bs=" << bs << "]\n";
  trl::ScopedTimer timer;

  const unsigned int N = 128;
  auto backend = Fixture::template make_backend<Scalar, bs>();
  auto op = Fixture::template make_diagonal<Scalar, bs>(N);

  const unsigned int nev = 16;
  std::vector<Scalar> exact_eigenvalues(nev);
  for (std::size_t i = 0; i < nev; ++i) exact_eigenvalues[i] = static_cast<Scalar>((N - i) * (N - i));

  return test_lanczos_convergence(backend, op, exact_eigenvalues, Scalar(1e-8));
}

/** @brief 1D Laplacian: exact eigenvalues are 2 - 2*cos(k*pi/(N+1)), largest first. */
template <class Fixture, class Scalar, unsigned int bs>
bool run_convergence_laplace()
{
  std::cout << "\n[Laplace 1D Test: bs=" << bs << "]\n";
  trl::ScopedTimer timer;

  const unsigned int N = 128;
  auto backend = Fixture::template make_backend<Scalar, bs>();
  auto op = Fixture::template make_laplace<Scalar, bs>(N);

  const unsigned int nev = 16;
  std::vector<Scalar> exact_eigenvalues(nev);
  const Scalar pi = std::acos(Scalar(-1));
  for (std::size_t i = 0; i < nev; ++i) exact_eigenvalues[i] = Scalar(2) - Scalar(2) * std::cos((N - i) * pi / (N + 1));

  return test_lanczos_convergence(backend, op, exact_eigenvalues, Scalar(1e-8));
}

template <class Fixture>
int run_convergence_suite()
{
  std::cout << "========================================\n";
  std::cout << "  Lanczos Convergence Tests (" << Fixture::name << ")\n";
  std::cout << "========================================\n";

  int num_failed = 0;

  std::cout << "\n--- Diagonal Eigenvalue Problem ---\n";
  if (!run_convergence_diagonal<Fixture, double, 1>()) num_failed++;
  if (!run_convergence_diagonal<Fixture, double, 2>()) num_failed++;
  if (!run_convergence_diagonal<Fixture, double, 4>()) num_failed++;
  if (!run_convergence_diagonal<Fixture, double, 8>()) num_failed++;

  std::cout << "\n--- Laplace 1D Eigenvalue Problem ---\n";
  if (!run_convergence_laplace<Fixture, double, 1>()) num_failed++;
  if (!run_convergence_laplace<Fixture, double, 2>()) num_failed++;
  if (!run_convergence_laplace<Fixture, double, 4>()) num_failed++;
  if (!run_convergence_laplace<Fixture, double, 8>()) num_failed++;

  std::cout << "\n========================================\n";
  std::cout << "Tests completed: " << (num_failed == 0 ? "ALL PASSED" : std::to_string(num_failed) + " FAILED") << "\n";
  std::cout << "========================================\n";

  return num_failed;
}
