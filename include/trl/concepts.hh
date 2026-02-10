#pragma once

#include <concepts>
#include <span>
#include <vector>

namespace trl {

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
concept BlockVectorView = requires(BV bv, const BV& other, const BV::MatrixBlockView& W) {
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
concept BlockMultiVector = requires(BMV bmv, std::size_t i) {
  typename BMV::Scalar;
  typename BMV::BlockView;
  typename BMV::BlockMatrix;

  { BMV::blocksize } -> std::convertible_to<unsigned int>;

  { bmv.block_view(i) } -> std::same_as<typename BMV::BlockView>;

  requires BlockVectorView<typename BMV::BlockView>;

  requires BlockMatrixConcept<typename BMV::BlockMatrix>;
};

/** @brief Concept for eigenvalue problems
 *
 *  An Eigenproblem bundles the operator, inner product, and small dense solver
 *  needed by the block Lanczos algorithm. It also provides factory methods for
 *  backend-specific multivectors and block matrices.
 *
 *  @par Requirements
 *  - types: Scalar, BlockMultivector (satisfies BlockMultiVector)
 *  - blocksize consistent with BlockMultivector::blocksize
 *  - apply: Y = A * X (or transformed operator)
 *  - dot: B = X^T * Y (or B-inner product)
 *  - orthonormalize: X <- X * R^{-1}, output R (upper triangular)
 *  - size: global problem dimension
 *  - create_multivector, create_blockmatrix: factory methods
 *  - solve_small_dense: solve projected problem, return converged count
 *  - get_current_eigenvalues, get_current_eigenvectors: latest Ritz data
 *  - get_eigenvalues_block: block of eigenvalues used for restart
 *
 *  @par Semantics
 *  For standard problems, dot computes X^T * Y. For generalized problems, dot
 *  may implement X^T * B * Y. The orthonormalize routine should leave R such
 *  that X_old = X_new * R, with R upper triangular.
 */
template <class EVP>
concept Eigenproblem =
    requires(EVP& evp, typename EVP::BlockMultivector::BlockView x, typename EVP::BlockMultivector::BlockMatrix::BlockView B, typename EVP::BlockMultivector::BlockMatrix B_mat,
             typename EVP::Scalar* eigvals, typename EVP::BlockMultivector::BlockMatrix& eigvecs) {
      typename EVP::Scalar;
      typename EVP::BlockMultivector;

      { EVP::blocksize } -> std::convertible_to<unsigned int>;
      { EVP::BlockMultivector::blocksize } -> std::convertible_to<unsigned int>;

      requires BlockMultiVector<typename EVP::BlockMultivector>;

      { evp.apply(x, x) } -> std::same_as<void>;

      { evp.dot(x, x, B) } -> std::same_as<void>;

      { evp.orthonormalize(x, B) } -> std::same_as<void>;

      { evp.size() } -> std::same_as<std::size_t>;

      { evp.create_multivector(std::size_t{}, std::size_t{}) } -> std::same_as<typename EVP::BlockMultivector>;

      { evp.create_blockmatrix(std::size_t{}, std::size_t{}) } -> std::same_as<typename EVP::BlockMultivector::BlockMatrix>;

      { evp.solve_small_dense(B_mat, B, std::size_t{}) } -> std::same_as<std::size_t>;

      { evp.get_current_eigenvalues() };
      { evp.get_current_eigenvectors() } -> std::same_as<const typename EVP::BlockMultivector::BlockMatrix&>;
      { evp.get_eigenvalues_block(std::size_t{}) } -> std::convertible_to<std::span<typename EVP::Scalar>>;
    } &&
    (
        EVP::blocksize == EVP::BlockMultivector::blocksize);

} // namespace trl
