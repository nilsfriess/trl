#pragma once

#include <trl/concepts.hh>
#include <trl/eigensolvers/lanczos.hh>

#include <iostream>
#include <memory>

#include "helpers.hh"
#include "test_helper.hh"

template <trl::BackendConcept B, trl::OperatorConcept<B> O>
bool test_lanczos_extend(B& backend, std::shared_ptr<O> op, typename B::Scalar tolerance, bool verbose)
{
  using Scalar = typename B::Scalar;
  constexpr auto bs = B::blocksize;
  auto N = op->size();

  std::cout << "Testing Lanczos relation, type = " << trl::type_str<Scalar>() << ", N = " << N << ", bs = " << bs << ": " << std::flush;

  trl::EigensolverParams params{.nev = 8, .ncv = 32, .max_restarts = 1000};
  trl::BlockLanczos lanczos(backend, op, params);

  auto V0 = lanczos.initial_block();
  trl::test::set_random(backend, V0);

  const unsigned int num_blocks = params.ncv / bs;
  lanczos.extend(0, num_blocks);

  backend.sync();

  auto& V = lanczos.get_basis();
  auto& T = lanczos.get_T();

  // Check if the basis is orthogonal
  bool passed = trl::check_orthogonality(backend, V, tolerance, verbose);

  // Check if the basis satisfies the Lanczos relation
  // Compute A*V (only for the first num_blocks blocks, not including the last extended block)
  auto AV = backend.make_multivector(N, params.ncv);
  for (unsigned int i = 0; i < num_blocks; ++i) op->apply(V.block_view(i), AV.block_view(i));

  backend.sync(); // Wait for all apply operations

  // Compute V*T (using only the first num_blocks blocks of V).
  // We only have a multiplication for blocks, so we need to do the full multiplication by hand.
  auto VT = backend.make_multivector(N, params.ncv);
  for (std::size_t j = 0; j < T.block_cols(); ++j) {
    auto VTj = VT.block_view(j);
    VTj.set_zero();
    for (std::size_t i = 0; i < T.block_cols(); ++i) {
      auto Tij = T.block_view(i, j);
      auto Vi = V.block_view(i);

      Vi.mult_add(Tij, VTj);
    }
  }

  backend.sync(); // Wait for all mult_add operations

  // Compute residual: AV - VT
  for (std::size_t i = 0; i < AV.blocks(); ++i) {
    auto AVi = AV.block_view(i);
    auto VTi = VT.block_view(i);
    AVi -= VTi;
  }

  backend.sync(); // Wait for all subtraction operations

  // Check norms of all blocks except the last
  Scalar max_error = 0;

  if (verbose) std::cout << "\n  Block norms of AV - VT:" << std::endl;

  for (unsigned int i = 0; i < num_blocks - 1; ++i) {
    auto norm = trl::test::norm(backend, AV.block_view(i));
    if (verbose) std::cout << "    Block " << i << ": " << norm << std::endl;
    if (norm > tolerance) {
      passed = false;
      max_error = std::max(max_error, norm);
    }
  }

  // Last block should equal V_{k+1} * beta
  // Compute residual term: V_{num_blocks} * beta
  auto& beta_mat = lanczos.get_beta();
  auto beta = beta_mat.block_view(0, 0);
  auto V_kplus1 = V.block_view(num_blocks);

  auto residual_term = backend.make_multivector(N, bs);
  auto residual_block = residual_term.block_view(0);
  V_kplus1.mult(beta, residual_block);

  backend.sync(); // Wait for mult operation

  // Check difference between (AV - VT)_{last} and V_{k+1} * beta^T
  auto last_block_view = AV.block_view(num_blocks - 1);
  last_block_view -= residual_block;

  backend.sync(); // Wait for subtraction

  auto last_block_error = trl::test::norm(backend, last_block_view);

  if (verbose) std::cout << "    Block " << (num_blocks - 1) << " (last) error: " << last_block_error << std::endl;

  if (last_block_error > tolerance) {
    passed = false;
    max_error = std::max(max_error, last_block_error);
  }

  if (passed) std::cout << "Passed." << std::endl;
  else std::cout << "Not passed. Max error: " << max_error << std::endl;

  return passed;
}

template <class Fixture, class Scalar, unsigned int bs>
bool run_extend_diagonal(bool verbose)
{
  const unsigned int N = 128;
  auto backend = Fixture::template make_backend<Scalar, bs>();
  auto op = Fixture::template make_diagonal<Scalar, bs>(N);

  return test_lanczos_extend(backend, op, Scalar(1e-8), verbose);
}

template <class Fixture>
int run_extend_suite(bool verbose = true)
{
  std::cout << "========================================\n";
  std::cout << "<<<<<<<<<   " << Fixture::name << " TEST   >>>>>>>>>\n";
  std::cout << "========================================\n";

  std::cout << "========================================\n";
  std::cout << "Testing with DiagonalEVP\n";
  std::cout << "========================================\n";

  int num_failed = 0;

  if (!run_extend_diagonal<Fixture, double, 1>(verbose)) num_failed++;
  if (!run_extend_diagonal<Fixture, double, 2>(verbose)) num_failed++;
  if (!run_extend_diagonal<Fixture, double, 4>(verbose)) num_failed++;
  if (!run_extend_diagonal<Fixture, double, 8>(verbose)) num_failed++;

  return num_failed;
}
