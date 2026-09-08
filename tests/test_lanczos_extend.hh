#pragma once

#include <trl/concepts.hh>
#include <trl/eigensolvers/lanczos.hh>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "helpers.hh"
#include "test_helper.hh"

/** @brief Checks the Lanczos relation A V = V T + v_{m} beta e_{m-1}^T.
 *
 *  Also reports the quantities the projected-T formulation rests on: the
 *  orthogonality of the basis, and the size of the entries of T outside the
 *  tridiagonal band, which the derivation says must be O(eps * ||A||) rather
 *  than the structural zeros they were when T was assembled from the
 *  recurrence.
 */
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
  const std::size_t ncv = params.ncv;

  lanczos.extend(0, num_blocks);
  backend.sync();

  auto& V = lanczos.get_basis();

  // Check if the basis is orthogonal
  bool passed = trl::check_orthogonality(backend, V, tolerance, verbose);

  // Only the upper triangle of T is computed (step i writes rows 0..i of column
  // i), so mirror it before using it as a matrix.
  std::vector<Scalar> T_host(ncv * ncv);
  {
    auto host = backend.host_block(lanczos.get_T(), trl::Access::Read);
    std::copy_n(host.data(), ncv * ncv, T_host.begin());
  }

  Scalar max_band = 0, max_offband = 0;
  for (std::size_t r = 0; r < ncv; ++r) {
    for (std::size_t c = r; c < ncv; ++c) {
      const auto value = std::abs(T_host[r * ncv + c]);
      if (c / bs - r / bs > 1) max_offband = std::max(max_offband, value);
      else max_band = std::max(max_band, value);
    }
  }

  if (verbose) std::cout << "    Max |T| off the tridiagonal band: " << max_offband << " (band scale " << max_band << ")" << std::endl;

  auto Tsym = backend.make_dense_matrix(ncv, ncv);
  {
    auto host = backend.host_block(Tsym, trl::Access::Write);
    for (std::size_t r = 0; r < ncv; ++r)
      for (std::size_t c = 0; c < ncv; ++c) host[r * ncv + c] = (r <= c) ? T_host[r * ncv + c] : T_host[c * ncv + r];
  }

  // Compute A*V (only for the first num_blocks blocks, not including the last extended block)
  auto AV = backend.make_multivector(N, params.ncv);
  for (unsigned int i = 0; i < num_blocks; ++i) op->apply(V.block_view(i), AV.block_view(i));

  // Compute V*T as a single panel product.
  auto VT = backend.make_multivector(N, params.ncv);
  V.panel_view(0, num_blocks).mult(trl::TransposeMode::NoTranspose, Tsym, VT.panel_view(0, num_blocks));

  // Residual: AV - VT
  AV.panel_view(0, num_blocks).subtract(VT.panel_view(0, num_blocks));
  backend.sync();

  Scalar max_error = 0;

  if (verbose) std::cout << "  Block norms of AV - VT:" << std::endl;

  for (unsigned int i = 0; i < num_blocks - 1; ++i) {
    auto norm = trl::test::norm(backend, AV.block_view(i));
    if (verbose) std::cout << "    Block " << i << ": " << norm << std::endl;
    if (norm > tolerance) {
      passed = false;
      max_error = std::max(max_error, norm);
    }
  }

  // The last block should equal V_{num_blocks} * beta.
  auto residual_term = backend.make_multivector(N, bs);
  auto residual_block = residual_term.block_view(0);
  V.block_view(num_blocks).mult(trl::TransposeMode::NoTranspose, lanczos.get_beta(), residual_block);

  auto last_block_view = AV.block_view(num_blocks - 1);
  last_block_view.subtract(residual_block);
  backend.sync();

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
