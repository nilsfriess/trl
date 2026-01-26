#pragma once

#include <trl/concepts.hh>
#include <trl/eigensolvers/lanczos.hh>

#include <memory>

#include "helpers.hh"

template <trl::Eigenproblem EVP, class TestHelper>
bool test_lanczos_extend(std::shared_ptr<EVP> evp, TestHelper& helper, typename EVP::Scalar tolerance, bool verbose)
{
  using Scalar = typename EVP::Scalar;
  constexpr auto bs = EVP::blocksize;
  auto N = evp->size();

  std::cout << "Testing Lanczos relation, type = " << trl::type_str<Scalar>() << ", N = " << N << ", bs = " << bs << ": ";

  // Use fewer columns than N to avoid Krylov subspace exhaustion
  // With ncv = N/2, we can do ncv/bs iterations without breakdown
  trl::EigensolverParams params{.nev = 24, .ncv = 64, .max_restarts = 1000};
  trl::BlockLanczos lanczos(evp, params);

  helper.sync();

  auto V0 = lanczos.initial_block();
  helper.set_random(V0);

  helper.sync();

  const unsigned int num_blocks = params.ncv / bs;
  lanczos.extend(0, num_blocks);

  helper.sync();

  auto& V = lanczos.get_basis();
  auto& T = lanczos.get_T();

  // Check if the basis is orthogonal
  bool passed = trl::check_orthogonality(*evp, helper, V, tolerance, verbose);

  // Check if the basis satisfies the Lanczos relation
  // Compute A*V (only for the first num_blocks blocks, not including the last extended block)
  auto AV = evp->create_multivector(N, params.ncv);
  for (unsigned int i = 0; i < num_blocks; ++i) evp->apply(V.block_view(i), AV.block_view(i));

  helper.sync();  // Wait for all apply operations

  // Compute V*T (using only the first num_blocks blocks of V).
  // We only have a multiplication for blocks, so we need to do the full multiplication by hand.
  auto VT = evp->create_multivector(N, params.ncv);
  for (std::size_t j = 0; j < T.block_cols(); ++j) {
    auto VTj = VT.block_view(j);
    VTj.set_zero();
    for (std::size_t i = 0; i < T.block_cols(); ++i) {
      auto Tij = T.block_view(i, j);
      auto Vi = V.block_view(i);

      Vi.mult_add(Tij, VTj);
    }
  }

  helper.sync();  // Wait for all mult_add operations

  // Compute residual: AV - VT
  for (std::size_t i = 0; i < AV.blocks(); ++i) {
    auto AVi = AV.block_view(i);
    auto VTi = VT.block_view(i);
    AVi -= VTi;
  }

  helper.sync();  // Wait for all subtraction operations

  // Check norms of all blocks except the last
  Scalar max_error = 0;

  if (verbose) std::cout << "\n  Block norms of AV - VT:" << std::endl;

  for (unsigned int i = 0; i < num_blocks - 1; ++i) {
    auto norm = helper.norm(AV.block_view(i));
    if (verbose) std::cout << "    Block " << i << ": " << norm << std::endl;
    if (norm > tolerance) {
      passed = false;
      max_error = std::max(max_error, norm);
    }
  }

  // Last block should equal V_{k+1} * beta
  // Compute residual term: V_{num_blocks} * beta
  auto& B = lanczos.get_B();
  auto beta = B.block_view(0, 0);
  auto V_kplus1 = V.block_view(num_blocks);

  auto residual_term = evp->create_multivector(N, bs);
  auto residual_block = residual_term.block_view(0);
  V_kplus1.mult(beta, residual_block);

  helper.sync();  // Wait for mult operation

  // Check difference between (AV - VT)_{last} and V_{k+1} * beta^T
  auto last_block_view = AV.block_view(num_blocks - 1);
  last_block_view -= residual_block;

  helper.sync();  // Wait for subtraction

  auto last_block_error = helper.norm(last_block_view);

  if (verbose) std::cout << "    Block " << (num_blocks - 1) << " (last) error: " << last_block_error << std::endl;

  if (last_block_error > tolerance) {
    passed = false;
    max_error = std::max(max_error, last_block_error);
  }

  if (passed) std::cout << "Passed." << std::endl;
  else std::cout << "Not passed. Max error: " << max_error << std::endl;

  return passed;

  std::cout << (passed ? "" : "not ") << "passed\n";
  return passed;
}
