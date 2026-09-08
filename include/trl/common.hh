#pragma once

#include "concepts.hh"

namespace trl {
/** @brief The Euclidean inner product, as an Operator mixin.
 *
 *  X may be a panel of any width, so this is the hook a generalized
 *  eigenproblem overrides to substitute a B-inner product: the eigensolver
 *  never reaches past op.dot for the coefficients it projects with.
 */
template <BackendConcept B>
struct EuclideanDot {
  void dot(typename B::Multivector::PanelView X, typename B::Multivector::BlockView Y, typename B::DenseMatrix R) const { X.dot(Y, R); }
};
} // namespace trl
