#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

#include "trl/concepts.hh"
#include "trl/eigensolvers/lanczos.hh"
#include "trl/eigensolvers/params.hh"
#include "trl/impl/gingko/multivector.hh"

#include <ginkgo/ginkgo.hpp>

#include "diagonal.hh"

int main(int argc, char* argv[])
{
  using Scalar = double;
  constexpr unsigned int bs = 1;
  std::size_t N = 256; // Matrix size

  using BMV = trl::ginkgo::BlockMultivector<Scalar, bs>;
  using EVP = DiagonalEigenproblem<Scalar, bs>;
  static_assert(trl::BlockMultiVector<BMV>);
  static_assert(trl::Eigenproblem<EVP>);

  auto exec = gko::OmpExecutor::create();

  auto evp = std::make_shared<EVP>(exec, N);

  trl::EigensolverParams params{
      .nev = 16,
      .ncv = 32,
  };

  trl::BlockLanczos lanczos(evp, params);

  // Initialise first block randomly
  {
    auto V0 = lanczos.get_basis().block_view(0);
    std::mt19937 gen(42);
    std::normal_distribution<Scalar> dist(0.0, 1.0);
    for (std::size_t i = 0; i < V0.rows(); ++i)
      for (std::size_t j = 0; j < V0.cols(); ++j) V0.data()->get_values()[i * V0.cols() + j] = dist(gen);
  }

  auto result = lanczos.solve();

  exec->synchronize();

  // Check convergence
  if (result.converged) { std::cout << "Solver converged after " << result.iterations << " iterations\n"; }
  else {
    std::cout << "Solver did NOT converge after " << result.iterations << " iterations\n";
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
    double exact = static_cast<double>(i + 1);
    double error = std::abs(computed - exact);
    max_error = std::max(max_error, error);
    std::cout << "  " << std::setw(5) << i << "    " << std::setw(10) << std::scientific << std::setprecision(6) << computed << "    " << std::setw(10) << std::scientific << std::setprecision(6)
              << exact << "    " << std::setw(10) << std::scientific << std::setprecision(2) << error << "\n";
  }
  std::cout << "\nMaximum error: " << std::scientific << std::setprecision(2) << max_error << "\n";

  return 0;
}
