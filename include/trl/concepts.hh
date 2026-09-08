#pragma once

#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <utility>

namespace trl {

enum class Access {
  Read,
  Write,
  ReadWrite,
};

enum class TransposeMode {
  Transpose,
  NoTranspose,
};

/** @brief Concept for block multivectors
 *
 *  A BlockMultivector stores a set of vectors grouped into fixed-size blocks.
 *  It is the primary data structure for Lanczos basis vectors.
 *
 *  The unit of work is the panel -- a view of `count` consecutive blocks -- and
 *  a single block is the count == 1 case, so BlockView and PanelView are the
 *  same type. Panel operations are what keep the number of kernel launches
 *  independent of the size of the basis.
 */
template <class BMV>
concept MultivectorConcept = requires(BMV bmv, std::size_t i, unsigned int first, unsigned int count) {
  typename BMV::Scalar;
  typename BMV::BlockView;
  typename BMV::PanelView;
  { BMV::blocksize } -> std::convertible_to<unsigned int>;

  { bmv.block_view(i) } -> std::same_as<typename BMV::BlockView>;
  { bmv.panel_view(first, count) } -> std::same_as<typename BMV::PanelView>;
  { bmv.blocks() } -> std::same_as<std::size_t>;
};

/** @brief Concept for backends
 *
 *  A backend names the storage types, allocates them, owns the host<->device
 *  boundary, and provides the handful of small dense kernels the eigensolver
 *  needs but cannot express through a panel.
 *
 *  DenseMatrix is a non-owning row-major view carrying a leading dimension;
 *  OwnedDenseMatrix is the allocation it views. Passing a DenseMatrix by value
 *  is how every operand of the small kernels is named.
 */
template <class B>
concept BackendConcept = requires(B& b, std::size_t n, unsigned int cols, unsigned int rows, typename B::DenseMatrix M, typename B::Scalar* status) {
  typename B::Scalar;
  typename B::Multivector;
  typename B::DenseMatrix;
  typename B::OwnedDenseMatrix;
  { B::blocksize } -> std::convertible_to<unsigned int>;

  { b.make_multivector(n, cols) } -> std::same_as<typename B::Multivector>;
  { b.make_dense_matrix(rows, cols) } -> std::same_as<typename B::OwnedDenseMatrix>;

  { b.sync() } -> std::same_as<void>;

  // Scoped host mirror. Reads on construction and/or writes back on destruction
  // according to the Access mode.
  { b.host_block(M, Access::ReadWrite) };
  { b.host_block(std::declval<typename B::Multivector::BlockView>(), Access::ReadWrite) };

  requires MultivectorConcept<typename B::Multivector>;
};

template <class O, class B>
concept OperatorConcept = BackendConcept<B> && requires(O& op, typename B::Multivector::BlockView x, typename B::Multivector::PanelView p, typename B::DenseMatrix R) {
  { op.apply(x, x) } -> std::same_as<void>;
  { op.dot(p, x, R) } -> std::same_as<void>;
  { op.size() } -> std::same_as<std::size_t>;
};

} // namespace trl
