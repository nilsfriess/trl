#pragma once

#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <span>

inline void todo_impl(const char* file, int line, const char* message)
{
  std::cerr << file << ":" << line << ": TODO: " << message << std::endl;
  std::abort();
}

#define TRL_TODO(message) todo_impl(__FILE__, __LINE__, message)

namespace trl {

enum class Access {
  Read,
  Write,
  ReadWrite,
};

/** @brief Concept for matrix block views
 *
 *  A MatrixBlockView represents a view into a small dense block (typically bs x bs)
 *  that lives inside a BlockMatrix. It is used for local block operations during
 *  Lanczos orthogonalization and for assembling the projected matrix.
 *
 *  @par Requirements
 *  - copy_from: copy entries from another block view
 *  - copy_from_transpose: copy the transpose of another block
 *  - set_zero: clear all entries
 *  - set_diagonal: write diagonal entries from a span of values
 *  - mult: compute C = this * B (block-matrix multiply)
 *
 *  @par Notes
 *  These operations are expected to be small, dense, and fast; no allocation
 *  should occur inside the view operations.
 */
template <class MBV>
concept MatrixBlockViewConcept = requires(MBV mbv, const MBV& other, std::span<typename MBV::EntryType> diag) {
  typename MBV::EntryType;

  { mbv.copy_from(other) } -> std::same_as<void>;
  { mbv.copy_from_transpose(other) } -> std::same_as<void>;
  { mbv.set_zero() } -> std::same_as<void>;
  { mbv.set_diagonal(diag) } -> std::same_as<void>;
  { mbv.mult(other, other) } -> std::same_as<void>;
};

/** @brief Concept for block matrices
 *
 *  A BlockMatrix stores a matrix as a grid of small dense blocks. It is used
 *  to represent the projected Lanczos matrix and small temporary blocks.
 *
 *  @par Requirements
 *  - block_rows, block_cols: number of block rows and columns
 *  - block_view: obtain a view into the (i, j) block
 *  - BlockView satisfies MatrixBlockViewConcept
 *
 *  @par Notes
 *  The block layout is implementation-defined, but block_view must behave like
 *  a lightweight, non-owning view.
 */
template <class BM>
concept BlockMatrixConcept = requires(BM bm, std::size_t i, std::size_t j) {
  typename BM::BlockView;

  { bm.block_rows() } -> std::same_as<std::size_t>;
  { bm.block_cols() } -> std::same_as<std::size_t>;
  { bm.block_view(i, j) } -> std::same_as<typename BM::BlockView>;

  requires MatrixBlockViewConcept<typename BM::BlockView>;
};

/** @brief Concept for block vector views
 *
 *  A BlockView is a view into a contiguous block of vectors inside a
 *  BlockMultivector. The Lanczos algorithm uses these views to build and
 *  orthogonalize block Krylov bases.
 *
 *  @par Requirements
 *  - set_zero, rows, cols, copy_from
 *  - operator-=: in-place subtraction
 *  - mult_add: other += this * W
 *  - mult: other = this * W
 *  - mult_transpose: other = this * W^T
 *  - subtract_product: this -= other * W
 *
 *  @par Notes
 *  Operations are expected to be dense, block-sized linear algebra kernels.
 */
template <class BV>
concept MultivectorBlockViewConcept = requires(BV bv, const BV& other, const BV::MatrixBlockView& W) {
  typename BV::EntryType;

  { bv.set_zero() } -> std::same_as<void>;
  { bv.rows() } -> std::same_as<std::size_t>;
  { bv.cols() } -> std::same_as<std::size_t>;
  { bv.copy_from(other) } -> std::same_as<void>;
  { bv -= other } -> std::same_as<BV&>;
  { bv.mult_add(W, other) } -> std::same_as<void>;
  { bv.mult(W, other) } -> std::same_as<void>;
  { bv.mult_transpose(W, other) } -> std::same_as<void>;
  { bv.subtract_product(other, W) } -> std::same_as<void>;
};

/** @brief Concept for block multivectors
 *
 *  A BlockMultivector stores a set of vectors grouped into fixed-size blocks.
 *  It is the primary data structure for Lanczos basis vectors.
 *
 *  @par Requirements
 *  - blocksize: compile-time block size
 *  - block_view: access a block view by block index
 *  - BlockView satisfies BlockVectorView
 *  - BlockMatrix satisfies BlockMatrixConcept
 */
template <class BMV>
concept MultivectorConcept = requires(BMV bmv, std::size_t i) {
  typename BMV::Scalar;
  typename BMV::BlockView;
  typename BMV::BlockMatrix;
  { BMV::blocksize } -> std::convertible_to<unsigned int>;

  { bmv.block_view(i) } -> std::same_as<typename BMV::BlockView>;
  { bmv.blocks() } -> std::same_as<std::size_t>;

  requires MultivectorBlockViewConcept<typename BMV::BlockView>;
  requires BlockMatrixConcept<typename BMV::BlockMatrix>;
};

/** @brief Concept for the scoped host mirror returned by Backend::host_block
 *
 *  Exposes the mirrored block as contiguous row-major storage of
 *  blocksize x blocksize entries. The handle is neither copyable nor movable:
 *  its lifetime is the transfer window.
 */
template <class H, class T>
concept HostBlockHandle = requires(H& h, const H& ch, std::size_t i) {
  { h.data() } -> std::same_as<T*>;
  { ch.data() } -> std::same_as<const T*>;
  { h[i] } -> std::same_as<T&>;
  { ch[i] } -> std::same_as<const T&>;
  { ch.size() } -> std::same_as<std::size_t>;
};

template <class B>
concept Backend = requires(B& b, std::size_t n, unsigned int cols, unsigned int br, unsigned int bc) {
  typename B::Scalar;
  typename B::Multivector;
  typename B::BlockMatrix;
  { B::blocksize } -> std::convertible_to<unsigned int>;

  { b.make_multivector(n, cols) } -> std::same_as<typename B::Multivector>;
  { b.make_blockmatrix(br, bc) } -> std::same_as<typename B::BlockMatrix>;

  { b.sync() } -> std::same_as<void>;

  // Scoped host mirror of a small block. Reads on construction and/or writes
  // back on destruction according to the Access mode, so the device-side value
  // is only guaranteed current once the handle has gone out of scope.
  typename B::HostBlock;
  { b.host_block(std::declval<typename B::BlockMatrix::BlockView>(), Access::ReadWrite) } -> std::same_as<typename B::HostBlock>;
  requires HostBlockHandle<typename B::HostBlock, typename B::Scalar>;

  // The same mirror over a multivector block, whose extent is only known at run time.
  typename B::HostVectors;
  { b.host_block(std::declval<typename B::Multivector::BlockView>(), Access::ReadWrite) } -> std::same_as<typename B::HostVectors>;
  requires HostBlockHandle<typename B::HostVectors, typename B::Scalar>;

  requires MultivectorConcept<typename B::Multivector>;
  requires BlockMatrixConcept<typename B::BlockMatrix>;
};

template <class O, class B>
concept Operator = Backend<B> && requires(O& op, typename B::Multivector::BlockView x, typename B::BlockMatrix::BlockView R) {
  { op.apply(x, x) } -> std::same_as<void>;
  { op.dot(x, x, R) } -> std::same_as<void>;
  { op.size() } -> std::same_as<std::size_t>;
};

} // namespace trl
