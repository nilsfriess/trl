#pragma once

#include <concepts>

namespace trl {

/** @brief Modified Gram-Schmidt reorthogonalization.
 *
 *  For each basis vector \f$ v_j \f$ in turn, projects out its component from
 *  \f$ v \f$ and updates \f$ v \f$ immediately before moving to \f$ v_{j+1} \f$.
 */
struct ModifiedGS {
  template <typename EVP, typename BMV>
  void operator()(EVP& evp, BMV& V, unsigned int count, typename BMV::BlockView V_next, typename BMV::BlockMatrix::BlockView tmp) const
  {
    for (unsigned int j = 0; j < count; ++j) {
      auto Vj = V.block_view(j);
      evp.dot(Vj, V_next, tmp);
      V_next.subtract_product(Vj, tmp);
    }
  }
};

/** @brief Concept for reorthogonalization strategies.
 *
 *  A strategy must be callable with (EVP&, BMV&, unsigned count, BlockView, BlockMatrixBlockView)
 *  and orthogonalize the BlockView against the first @p count blocks of the basis.
 */
template <typename R, typename EVP, typename BMV>
concept ReorthogonalizationStrategy = requires(R r, EVP& evp, BMV& V, unsigned int count, typename BMV::BlockView v, typename BMV::BlockMatrix::BlockView tmp) {
  { r(evp, V, count, v, tmp) } -> std::same_as<void>;
};

} // namespace trl
