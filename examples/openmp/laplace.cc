#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <ratio>
#include <type_traits>

#include "trl/eigensolvers/lanczos.hh"
#include "trl/eigensolvers/params.hh"
#include "trl/impl/openmp/evp_base.hh"

using namespace trl::openmp;

template <typename Scalar, unsigned int blocksize>
void run_case(std::size_t n, const trl::EigensolverParams& params, int runs)
{
  using EVP = EVPBase<Scalar, blocksize>;
  auto evp = std::make_shared<EVP>(n);

  std::cout << "\n=== Scalar=" << (std::is_same_v<Scalar, float> ? "float" : "double") << ", blocksize=" << blocksize << " ===\n";

  trl::BlockLanczos lanczos(evp, params);

  auto V0 = lanczos.initial_block();
  std::mt19937 rng(42);
  std::uniform_real_distribution<Scalar> dist(Scalar(-1), Scalar(1));
  for (std::size_t i = 0; i < V0.rows() * V0.cols(); ++i) V0.data[i] = dist(rng);

  auto result = lanczos.solve();

  std::cout << "Converged: " << result.converged << "\n";
  std::cout << "Iterations: " << result.iterations << "\n";
  std::cout << "Operator applications: " << result.n_op_apply << "\n\n";

  const auto& eigenvalues = evp->get_current_eigenvalues();
  std::cout << "Computed " << eigenvalues.size() << " eigenvalues:\n";
  for (std::size_t i = 0; i < std::min(eigenvalues.size(), 16UL); ++i) std::cout << "  λ[" << i << "] = " << eigenvalues[i] << "\n";

  std::cout << "Measuring performance... " << std::flush;
  std::chrono::duration<double, std::milli> total_time{0};
  for (int run = 0; run < runs; ++run) {
    trl::BlockLanczos lanczos2(evp, params);

    auto V0 = lanczos2.initial_block();
    std::mt19937 rng(42);
    std::uniform_real_distribution<Scalar> dist(Scalar(-1), Scalar(1));
    for (std::size_t i = 0; i < V0.rows() * V0.cols(); ++i) V0.data[i] = dist(rng);

    auto start = std::chrono::steady_clock::now();
    lanczos2.solve();
    auto end = std::chrono::steady_clock::now();

    total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  }
  const double avg_ms = total_time.count() / runs;
  std::cout << "done. Avg solve time: " << avg_ms << " ms (total " << total_time.count() << " ms over " << runs << " runs)\n";
}

int main()
{
  std::size_t n = 4096;
  trl::EigensolverParams params{.nev = 32, .ncv = 128, .max_restarts = 50};
  int runs = 5;

  run_case<float, 1>(n, params, runs);
  run_case<float, 2>(n, params, runs);
  run_case<float, 4>(n, params, runs);
  run_case<float, 8>(n, params, runs);
  run_case<float, 16>(n, params, runs);

  run_case<double, 1>(n, params, runs);
  run_case<double, 2>(n, params, runs);
  run_case<double, 4>(n, params, runs);
  run_case<double, 8>(n, params, runs);
  run_case<double, 16>(n, params, runs);
}
