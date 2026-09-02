#pragma once

#include "concepts.hh"

namespace trl {
template <Backend B>
struct EuclideanDot {
  void dot(typename B::Multivector::BlockView X, typename B::Multivector::BlockView Y, typename B::BlockMatrix::BlockView R) const { X.dot(Y, R); }
};
} // namespace trl
