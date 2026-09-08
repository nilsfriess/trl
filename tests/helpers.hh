#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>

#include <trl/concepts.hh>
#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>

namespace trl {
class ScopedTimer {
public:
  ScopedTimer() { start = std::chrono::high_resolution_clock::now(); }

  ~ScopedTimer()
  {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Elapsed: " << duration << " ms\n";
  }

private:
  std::chrono::high_resolution_clock::time_point start;
};

template <typename Scalar>
std::string type_str()
{
  if constexpr (std::is_same_v<Scalar, double>) return "double";
  else if constexpr (std::is_same_v<Scalar, float>) return "float";
  else return "unknown";
}

/** @brief Verifies that every block of V is orthonormal and mutually orthogonal.
 *
 *  One panel dot per block gives the whole column of the Gram matrix, so the
 *  check costs V.blocks() kernels and as many transfers rather than the
 *  quadratic number of block dots it used to.
 */
template <trl::BackendConcept B>
bool check_orthogonality(B& backend, typename B::Multivector& V, typename B::Scalar tolerance, bool verbose)
{
  using Scalar = typename B::Scalar;
  constexpr auto bs = B::blocksize;

  if (verbose) std::cout << "  Checking orthogonality of V blocks..." << std::endl;

  const auto nb = static_cast<unsigned int>(V.blocks());
  auto G = backend.make_dense_matrix(nb * bs, bs);
  auto Vp = V.panel_view(0, nb);

  Scalar max_offdiag = 0;
  Scalar max_diag_error = 0;
  int max_offdiag_i = -1, max_offdiag_j = -1;

  for (unsigned int i = 0; i < nb; ++i) {
    Vp.dot(V.block_view(i), G);

    auto host = backend.host_block(G, Access::Read);
    for (unsigned int j = 0; j < nb; ++j) {
      for (unsigned int r = 0; r < bs; ++r) {
        for (unsigned int c = 0; c < bs; ++c) {
          const auto value = host[(j * bs + r) * bs + c];

          if (i == j and r == c) max_diag_error = std::max(max_diag_error, std::abs(value - Scalar(1)));
          else if (std::abs(value) > max_offdiag) {
            max_offdiag = std::abs(value);
            max_offdiag_i = static_cast<int>(i);
            max_offdiag_j = static_cast<int>(j);
          }
        }
      }
    }
  }

  if (verbose) {
    std::cout << "    Max diagonal error: " << max_diag_error << std::endl;
    std::cout << "    Max off-block error: " << max_offdiag;
    if (max_offdiag_i >= 0) std::cout << " (blocks " << max_offdiag_i << "," << max_offdiag_j << ")";
    std::cout << std::endl;
  }

  return max_offdiag < tolerance and max_diag_error < tolerance;
}
} // namespace trl
