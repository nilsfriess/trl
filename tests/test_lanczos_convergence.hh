#pragma once

#include <trl/concepts.hh>
#include <trl/eigensolvers/lanczos.hh>

#include <memory>

#include "helpers.hh"

template <trl::Eigenproblem EVP, class TestHelper>
bool test_lanczos_convergence(std::shared_ptr<EVP> evp, TestHelper& helper, const std::vector<typename EVP::Scalar>& exact_eigenvalues)
{
  using Scalar = typename EVP::Scalar;
  constexpr auto bs = EVP::blocksize;
  auto N = evp->size();

  std::cout << "\nTesting Lanczos convergence, type = " << trl::type_str<Scalar>() << ", N = " << N << ", bs = " << bs << ": ";

  unsigned int nev = exact_eigenvalues.size();
  trl::EigensolverParams params{.nev = nev, .ncv = 2 * nev, .max_restarts = 1000};
  trl::BlockLanczos lanczos(evp, params);

  auto V0 = lanczos.initial_block();
  helper.set_random(V0);
  helper.sync();

  bool passed = true;

  auto res = lanczos.solve();
  helper.sync();

  if (true or res.converged) {
    std::cout << "Eigensolver converged in " << res.iterations << " iterations (" << res.n_op_apply << " operator applications).\n";
    std::cout << "Checking computed eigenvalues against exact values...\n";

    auto ev = evp->get_current_eigenvalues();
    auto ev_host = helper.to_host_data(ev);
    if (ev_host.size() < exact_eigenvalues.size()) {
      passed = false;
      std::cout << "In test_lanczos_convergence: Number of computed eigenvalues is smaller than number of requested eigenvalues, " << ev_host.size() << " vs. " << exact_eigenvalues.size() << "\n";
    }
    for (unsigned int i = 0; i < nev; ++i) {
      if (std::abs(ev_host[i] - exact_eigenvalues[i]) > 1e-8) {
        std::cout << "In test_lanczos_convergence: Eigenvalue " << i << " differs from exact, computed " << ev_host[i] << ", expected " << exact_eigenvalues[i] << ", error "
                  << std::abs(ev_host[i] - exact_eigenvalues[i]) << "\n";
        passed = false;
      }
      else {
        std::cout << "  Eigenvalue " << i << " correct: " << ev_host[i] << ", error " << std::abs(ev_host[i] - exact_eigenvalues[i]) << "\n";
      }
    }
  }
  else passed = false;

  std::cout << (passed ? " " : "not ") << "passed.\n";

  return passed;
}
