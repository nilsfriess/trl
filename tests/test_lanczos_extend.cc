#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include <trl/concepts.hh>
#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>

#include "../examples/diagonal.hh"
#include "../examples/laplace.hh"

template <typename Scalar>
std::string type_str()
{
  if constexpr (std::is_same_v<Scalar, double>) return "double";
  else if constexpr (std::is_same_v<Scalar, float>) return "float";
  else return "unknown";
}

// Helper to check for NaN/Inf in data
template <typename T>
bool check_nan_inf(T* data, std::size_t size, const std::string& name, bool verbose = false)
{
  for (std::size_t i = 0; i < size; ++i) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      if (verbose) std::cout << "  NaN/Inf detected in " << name << " at index " << i << ", value=" << data[i] << std::endl;
      return false;
    }
  }
  return true;
}

// Helper to verify orthogonality of V blocks
template <trl::Eigenproblem EVP>
void check_orthogonality(std::shared_ptr<EVP> evp, typename EVP::BlockMultivector& V, unsigned int num_blocks, bool verbose)
{
  using Scalar = typename EVP::Scalar;
  constexpr auto bs = EVP::blocksize;

  if (!verbose) return;

  std::cout << "  Checking orthogonality of V blocks..." << std::endl;
  Scalar max_offdiag = 0;
  Scalar max_diag_error = 0;

  auto temp = evp->create_blockmatrix(1, 1);
  auto temp_block = temp.block_view(0, 0);

  for (unsigned int i = 0; i < num_blocks; ++i) {
    auto Vi = V.block_view(i);

    // Check V_i^T * V_i = I
    Vi.dot(Vi, temp_block);
    for (unsigned int r = 0; r < bs; ++r) {
      for (unsigned int c = 0; c < bs; ++c) {
        Scalar val = temp_block.data[r * bs + c];
        if (r == c) max_diag_error = std::max(max_diag_error, std::abs(val - Scalar(1.0)));
        else max_offdiag = std::max(max_offdiag, std::abs(val));
      }
    }

    // Check V_i^T * V_j ≈ 0 for j < i
    for (unsigned int j = 0; j < i; ++j) {
      auto Vj = V.block_view(j);
      Vi.dot(Vj, temp_block);
      for (unsigned int k = 0; k < bs * bs; ++k) max_offdiag = std::max(max_offdiag, std::abs(temp_block.data[k]));
    }
  }

  std::cout << "    Max diagonal error: " << max_diag_error << std::endl;
  std::cout << "    Max off-block error: " << max_offdiag << std::endl;
}

// Test the Lanczos relation: A*V = V*T + V_{k+1}*beta^T*e_k^T
template <trl::Eigenproblem EVP>
bool test_lanczos_relation(std::shared_ptr<EVP> evp, bool verbose, typename EVP::Scalar tol = 1e-8)
{
  using Scalar = typename EVP::Scalar;
  constexpr auto bs = EVP::blocksize;
  auto N = evp->size();

  std::cout << "Testing Lanczos relation, type = " << type_str<Scalar>() << ", N = " << N << ", bs = " << bs << ": ";

  // Adjust ncv to ensure ncv/bs gives enough blocks
  trl::EigensolverParams params{.nev = 8, .ncv = 16};
  trl::BlockLanczos lanczos(evp, params);

  auto V0 = lanczos.initial_block();

  std::mt19937 rng(42);
  std::uniform_real_distribution<typename EVP::Scalar> dist(-1, 1);
  for (std::size_t i = 0; i < V0.rows() * V0.cols(); ++i) V0.data[i] = dist(rng);

  const unsigned int num_blocks = params.ncv / bs;
  lanczos.extend(0, num_blocks);

  auto& V = lanczos.get_basis();
  auto& T = lanczos.get_T();

  // Check for NaN/Inf in V and T
  if (verbose) {
    bool v_ok = check_nan_inf(V.block_view(0).data, N * params.ncv, "V", verbose);
    bool t_ok = check_nan_inf(T.block_view(0, 0).data, num_blocks * num_blocks * bs * bs, "T", verbose);
    if (!v_ok || !t_ok) {
      std::cout << "NaN/Inf detected! Test invalid." << std::endl;
      return false;
    }
  }

  // Check orthogonality
  check_orthogonality(evp, V, num_blocks, verbose);

  if (verbose) {
    std::cout << "  Dimensions: num_blocks=" << num_blocks << ", ncv=" << params.ncv << ", T: " << T.block_rows() << "x" << T.block_cols() << std::endl;

    // Print T matrix structure (Frobenius norms of each block)
    std::cout << "  T matrix block norms:" << std::endl;
    for (unsigned int i = 0; i < num_blocks; ++i) {
      std::cout << "    Row " << i << ": ";
      for (unsigned int j = 0; j < num_blocks; ++j) {
        auto Tij = T.block_view(i, j);
        Scalar norm = 0;
        for (unsigned int k = 0; k < bs * bs; ++k) norm += Tij.data[k] * Tij.data[k];
        norm = std::sqrt(norm);
        std::cout << norm << " ";
      }
      std::cout << std::endl;
    }
  }

  // Compute AV (only for the first num_blocks blocks, not including the last extended block)
  auto AV = evp->create_multivector(N, params.ncv);
  for (unsigned int i = 0; i < num_blocks; ++i) evp->apply(V.block_view(i), AV.block_view(i));

  // Compute VT (using only the first num_blocks blocks of V)
  auto VT = evp->create_multivector(N, params.ncv);
  V.mult(T, VT, num_blocks - 1);

  if (verbose) {
    std::cout << "  VT block norms:" << std::endl;
    for (unsigned int i = 0; i < num_blocks; ++i) std::cout << "    VT[" << i << "]: " << VT.block_view(i).norm() << std::endl;
    std::cout << "  AV block norms:" << std::endl;
    for (unsigned int i = 0; i < num_blocks; ++i) std::cout << "    AV[" << i << "]: " << AV.block_view(i).norm() << std::endl;
  }

  // Compute residual: AV - VT
  AV -= VT;

  // Check norms of all blocks except the last
  bool passed = true;
  Scalar max_error = 0;

  if (verbose) std::cout << "\n  Block norms of AV - VT:" << std::endl;

  for (unsigned int i = 0; i < num_blocks - 1; ++i) {
    auto norm = AV.block_view(i).norm();
    if (verbose) std::cout << "    Block " << i << ": " << norm << std::endl;
    if (norm > tol) {
      passed = false;
      max_error = std::max(max_error, norm);
    }
  }

  // Last block should equal V_{k+1} * beta^T
  // Compute residual term: V_{num_blocks} * beta^T
  auto& B = lanczos.get_B();
  auto beta = B.block_view(0, 0);
  auto V_kplus1 = V.block_view(num_blocks);

  auto residual_term = evp->create_multivector(N, bs);
  auto residual_block = residual_term.block_view(0);
  V_kplus1.mult(beta, residual_block);

  // Check difference between (AV - VT)_{last} and V_{k+1} * beta^T
  auto last_block_view = AV.block_view(num_blocks - 1);
  last_block_view -= residual_block;
  auto last_block_error = last_block_view.norm();

  if (verbose) std::cout << "    Block " << (num_blocks - 1) << " (last) error: " << last_block_error << std::endl;

  if (last_block_error > tol) {
    passed = false;
    max_error = std::max(max_error, last_block_error);
  }

  if (passed) std::cout << "Passed." << std::endl;
  else std::cout << "Not passed. Max error: " << max_error << std::endl;

  return passed;
}

int main()
{
  sycl::queue q;
  bool verbose = false;
  int num_failed = 0;

  // Test with DiagonalEVP
  std::cout << "========================================\n";
  std::cout << "Testing with DiagonalEVP\n";
  std::cout << "========================================\n";

  {
    std::cout << "Block size 1:\n";
    const unsigned int N = 32;
    using EVP = DiagonalEVP<double, 1>;
    auto evp = std::make_shared<EVP>(q, N);
    if (!test_lanczos_relation(evp, verbose)) num_failed++;
    std::cout << std::endl;
  }

  {
    std::cout << "Block size 2:\n";
    const unsigned int N = 32;
    using EVP = DiagonalEVP<double, 2>;
    auto evp = std::make_shared<EVP>(q, N);
    if (!test_lanczos_relation(evp, verbose)) num_failed++;
    std::cout << std::endl;
  }

  {
    std::cout << "Block size 4:\n";
    const unsigned int N = 128;
    using EVP = DiagonalEVP<double, 4>;
    auto evp = std::make_shared<EVP>(q, N);
    if (!test_lanczos_relation(evp, verbose)) num_failed++;
    std::cout << std::endl;
  }

  // Test with OneDLaplaceEVP
  std::cout << "========================================\n";
  std::cout << "Testing with OneDLaplaceEVP\n";
  std::cout << "========================================\n";

  {
    std::cout << "Block size 1:\n";
    const unsigned int N = 32;
    using EVP = trl::OneDLaplaceEVP<double, 1>;
    auto evp = std::make_shared<EVP>(q, N);
    if (!test_lanczos_relation(evp, verbose)) num_failed++;
    std::cout << std::endl;
  }

  {
    std::cout << "Block size 2:\n";
    const unsigned int N = 32;
    using EVP = trl::OneDLaplaceEVP<double, 2>;
    auto evp = std::make_shared<EVP>(q, N);
    if (!test_lanczos_relation(evp, verbose)) num_failed++;
    std::cout << std::endl;
  }

  {
    std::cout << "Block size 4:\n";
    const unsigned int N = 128;
    using EVP = trl::OneDLaplaceEVP<double, 4>;
    auto evp = std::make_shared<EVP>(q, N);
    if (!test_lanczos_relation(evp, verbose)) num_failed++;
    std::cout << std::endl;
  }

  std::cout << "========================================\n";
  if (num_failed == 0) {
    std::cout << "All tests passed!\n";
    return 0;
  }
  else {
    std::cout << num_failed << " test(s) failed!\n";
    return 1;
  }
}
