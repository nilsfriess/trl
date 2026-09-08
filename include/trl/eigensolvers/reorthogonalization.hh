#pragma once

#include "trl/concepts.hh"

namespace trl {

/** @brief Reorthogonalization strategies.
 *
 *  A strategy orthogonalizes the block @p w against the panel @p Vp and leaves
 *  the total projection Vp^T w in @p h.
 *
 *  @p scratch is a buffer of at least the shape of @p h, owned by the caller.
 */

/** @brief Classical twice-is-enough Gram-Schmidt. */
struct ClassicalGS2 {
  template <class Op, class Backend, class Panel, class Dense>
  void operator()(Op& op, Backend& backend, Panel Vp, Panel w, Dense h, Dense scratch) const
  {
    const auto pass = [&](auto& hh) {
      op.dot(Vp, w, hh);
      w.subtract_product(TransposeMode::NoTranspose, Vp, hh);
    };

    auto h2 = scratch.block(0, 0, h.rows(), h.cols());
    pass(h);
    pass(h2);
    h.add(h2);
  }
};

/** @brief Modified Gram-Schmidt. */
struct ModifiedGS {
  template <class Op, class Backend, class Panel, class Dense>
  void operator()(Op& op, Backend& backend, Panel Vp, Panel w, Dense h, Dense) const
  {
    constexpr unsigned int bs = Panel::blocksize;

    for (unsigned int j = 0; j < Vp.blocks(); ++j) {
      auto Vj = Vp.block(j);
      auto hj = h.block(j * bs, 0, bs, bs);
      op.dot(Vj, w, hj);
      w.subtract_product(TransposeMode::NoTranspose, Vj, hj);
    }
  }
};

/** @brief Concept for reorthogonalization strategies. */
template <class R, class Op, class Backend, class Panel, class Dense>
concept ReorthogonalizationStrategy = requires(R r, Op& op, Backend& backend, Panel Vp, Panel w, Dense h, Dense scratch) {
  { r(op, backend, Vp, w, h, scratch) } -> std::same_as<void>;
};

} // namespace trl
