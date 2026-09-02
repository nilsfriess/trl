#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

#include <trl/concepts.hh>

namespace trl::test {

/** @brief Fills a multivector block with normally distributed values.
 *
 *  The generator is seeded per call, so a given @p seed always produces the same
 *  block regardless of backend.
 */
template <trl::Backend B>
void set_random(B& backend, typename B::Multivector::BlockView V, std::uint_fast32_t seed = std::mt19937::default_seed)
{
  std::mt19937 rng(seed);
  std::normal_distribution<typename B::Scalar> dist;

  auto host = backend.host_block(V, Access::Write);
  std::generate_n(host.data(), host.size(), [&]() { return dist(rng); });
}

/** @brief Euclidean norm of a multivector block. */
template <trl::Backend B>
typename B::Scalar norm(B& backend, typename B::Multivector::BlockView V)
{
  auto host = backend.host_block(V, Access::Read);

  typename B::Scalar sum = 0;
  for (std::size_t i = 0; i < host.size(); ++i) sum += host[i] * host[i];
  return std::sqrt(sum);
}

} // namespace trl::test
