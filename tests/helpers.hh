#pragma once

#include <chrono>
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

// Helper to verify orthogonality of V blocks
template <trl::Eigenproblem EVP, class TestHelper>
bool check_orthogonality(EVP& evp, TestHelper& helper, typename EVP::BlockMultivector& V, typename EVP::Scalar tolerance, bool verbose)
{
  using Scalar = typename EVP::Scalar;
  constexpr auto bs = EVP::blocksize;

  if (verbose) std::cout << "  Checking orthogonality of V blocks..." << std::endl;
  Scalar max_offdiag = 0;
  Scalar max_diag_error = 0;

  auto temp = evp.create_blockmatrix(1, 1);
  auto temp_block = temp.block_view(0, 0);

  for (unsigned int i = 0; i < V.blocks(); ++i) {
    auto Vi = V.block_view(i);

    // Check V_i^T * V_i = I
    Vi.dot(Vi, temp_block);
    auto temp_block_host = helper.to_host_data(temp_block);

    for (unsigned int r = 0; r < bs; ++r) {
      for (unsigned int c = 0; c < bs; ++c) {
        auto val = temp_block_host[r * bs + c];
        if (r == c) max_diag_error = std::max(max_diag_error, std::abs(val - Scalar(1.0)));
        else max_offdiag = std::max(max_offdiag, std::abs(val));
      }
    }

    // Check V_i^T * V_j ≈ 0 for j < i
    for (unsigned int j = 0; j < i; ++j) {
      auto Vj = V.block_view(j);
      Vi.dot(Vj, temp_block);
      temp_block_host = helper.to_host_data(temp_block);
      for (unsigned int k = 0; k < bs * bs; ++k) max_offdiag = std::max(max_offdiag, std::abs(temp_block_host[k]));
    }
  }

  if (verbose) {
    std::cout << "    Max diagonal error: " << max_diag_error << std::endl;
    std::cout << "    Max off-block error: " << max_offdiag << std::endl;
  }

  return max_offdiag < tolerance and max_diag_error < tolerance;
}
} // namespace trl
