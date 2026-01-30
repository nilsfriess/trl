#pragma once

#include <concepts>
#include <vector>

namespace trl {

/** @brief Concept for matrix block views
 *
 *  A MatrixBlockView represents a view into a small dense matrix block.
 *  It provides operations for copying and transposing data.
 */
template <class MBV>
concept MatrixBlockViewConcept = requires(MBV mbv, const MBV& other) {
  typename MBV::EntryType;

  /** @brief Copy data from another matrix block view */
  { mbv.copy_from(other) } -> std::same_as<void>;

  /** @brief Copy and transpose data from another matrix block view */
  { mbv.copy_from_transpose(other) } -> std::same_as<void>;

  /** @brief Zero the given block */
  { mbv.set_zero() } -> std::same_as<void>;

  /** @brief Multiply by another block view and store in a block view */
  { mbv.mult(other, other) } -> std::same_as<void>;

  // TODO: Make this work
  // /** @brief Sets the diagonal */
  // { mbv.set_diagonal(...) } -> std::same_as<void>;
};

/** @brief Concept for block matrices
 *
 *  A BlockMatrix is a matrix organized into small dense blocks.
 *  It is typically used for the projected matrix T in Lanczos/Arnoldi methods.
 */
template <class BM>
concept BlockMatrixConcept = requires(BM bm, std::size_t i, std::size_t j) {
  typename BM::BlockView;

  /** @brief Number of block rows */
  { bm.block_rows() } -> std::same_as<std::size_t>;

  /** @brief Number of block columns */
  { bm.block_cols() } -> std::same_as<std::size_t>;

  /** @brief Access a view to the (i, j)-th block */
  { bm.block_view(i, j) } -> std::same_as<typename BM::BlockView>;

  /** @brief BlockView must satisfy the MatrixBlockView concept */
  requires MatrixBlockViewConcept<typename BM::BlockView>;
};

/** @brief Concept for block vector views
 *
 *  A BlockView represents a view into a contiguous block of vectors within a
 *  BlockMultivector. It provides operations for linear algebra without owning
 *  the underlying data.
 */
template <class BV>
concept BlockVectorView = requires(BV bv, const BV& other, const BV::MatrixBlockView& W) {
  typename BV::EntryType;

  /** @brief Zeros the given block */
  { bv.set_zero() } -> std::same_as<void>;

  /** @brief Number of rows in the block vector */
  { bv.rows() } -> std::same_as<std::size_t>;

  /** @brief Number of columns (i.e., number of vectors in the block) */
  { bv.cols() } -> std::same_as<std::size_t>;

  /** @brief Copy data from another block vector
   *
   *  Copies the data from source into this view. Both views must have
   *  the same dimensions. This copies the underlying data, not the view itself.
   */
  { bv.copy_from(other) } -> std::same_as<void>;

  /** @brief Subtract another block vector: this -= B
   *
   *  Performs element-wise subtraction, modifying this block vector in-place.
   */
  { bv -= other } -> std::same_as<BV&>;

  /** @brief Computes this += W * Y, where W is a small matrix and Y is another block view */
  { bv.mult_add(W, other) } -> std::same_as<void>;

  /** @brief Computes Y = this * W, where W is a small matrix and Y is another block view */
  { bv.mult(W, other) } -> std::same_as<void>;
};

/** @brief Concept for block multivectors
 *
 *  A BlockMultivector is a collection of vectors organized into blocks,
 *  each containing a fixed number of vectors (the blocksize).
 */
template <class BMV>
concept BlockMultiVector = requires(BMV bmv, std::size_t i) {
  typename BMV::Scalar;
  typename BMV::BlockView;
  typename BMV::BlockMatrix;

  /** @brief Compile-time constant: number of vectors per block */
  { BMV::blocksize } -> std::convertible_to<unsigned int>;

  /** @brief Access a view to the i-th block of vectors
   *
   *  Returns a BlockView providing access to vectors [i*blocksize, (i+1)*blocksize).
   */
  { bmv.block_view(i) } -> std::same_as<typename BMV::BlockView>;

  /** @brief BlockView must satisfy the BlockVectorView concept */
  requires BlockVectorView<typename BMV::BlockView>;

  /** @brief BlockMatrix must satisfy the BlockMatrix concept */
  requires BlockMatrixConcept<typename BMV::BlockMatrix>;
};

template <class EVP>
concept Eigenproblem =
    requires(EVP& evp, typename EVP::BlockMultivector::BlockView x, typename EVP::BlockMultivector::BlockMatrix::BlockView B, typename EVP::BlockMultivector::BlockMatrix B_mat,
             typename EVP::Scalar* eigvals, typename EVP::BlockMultivector::BlockMatrix& eigvecs) {
      typename EVP::BlockMultivector;

      { EVP::blocksize } -> std::convertible_to<unsigned int>;
      { EVP::BlockMultivector::blocksize } -> std::convertible_to<unsigned int>;

      /** @brief BlockMultivector must satisfy the BlockMultiVector concept */
      requires BlockMultiVector<typename EVP::BlockMultivector>;

      /** @brief Application of the operator that defines the eigenvalue problem
       *
       *  In a standard eigenvalue problem Ax = λx, this computes Y = AX.
       *  For a generalized eigenvalue problem Ax = λBx, this could implement
       *  a transformed operator, e.g., the shift-invert Y = (A - σB)^{-1}BX.
       */
      { evp.apply(x, x) } -> std::same_as<void>;

      /** @brief Compute the inner product matrix between two block vectors
       *
       *  For a standard eigenvalue problem Ax = λx, this computes the
       *  Euclidean inner product B = X^T Y.
       *  For a generalized eigenvalue problem Ax = λBx, this computes the
       *  B-inner product B = X^T B Y.
       */
      { evp.dot(x, x, B) } -> std::same_as<void>;

      /** @brief Orthonormalize a block vector
       *
       *  Computes V_new and R such that:
       *  - V_old = V_new * R  (equivalently: V_new = V_old * R^{-1})
       *  - V_new is orthonormal: V_new^T * V_new = I for standard problems,
       *    or V_new^T * B * V_new = I for generalized problems
       *
       *  Typically implemented using Cholesky QR:
       *  1. Compute G = V^T * V (or G = V^T * B * V for generalized problems)
       *  2. Compute Cholesky factorization: G = R^T * R
       *  3. Solve V_new = V * R^{-1}
       *
       *  @param V Block vector to orthonormalize (modified in place)
       *  @param R Output: upper triangular Cholesky factor (bs × bs matrix)
       */
      { evp.orthonormalize(x, B) } -> std::same_as<void>;

      /** @brief Return the dimension of the eigenvalue problem
       *
       *  Returns the number of rows in the operator matrix A (or the dimension
       *  of the vector space in which eigenvectors live).
       */
      { evp.size() } -> std::same_as<std::size_t>;

      /** @brief Factory method to create a BlockMultivector
       *
       *  Creates a multivector with the specified dimensions, encapsulating
       *  any backend-specific details (e.g., SYCL queues, GPU contexts, memory allocation).
       *
       *  @param rows Number of rows (dimension of each vector)
       *  @param cols Total number of columns (must be divisible by blocksize)
       */
      { evp.create_multivector(std::size_t{}, std::size_t{}) } -> std::same_as<typename EVP::BlockMultivector>;

      /** @brief Factory method to create a BlockMatrix
       *
       *  Creates a block-structured matrix with the specified block dimensions,
       *  encapsulating backend-specific details for small dense matrices used
       *  in the algorithm.
       *
       *  @param block_rows Number of block rows
       *  @param block_cols Number of block columns
       */
      { evp.create_blockmatrix(std::size_t{}, std::size_t{}) } -> std::same_as<typename EVP::BlockMultivector::BlockMatrix>;

      /** @brief Solve the small dense eigenvalue problem for the projected matrix
       *
       *  Computes eigenvalues and eigenvectors of the block tridiagonal matrix T.
       *  The output data is supposed to be managed by the EVP object.
       *
       *  @param T The block tridiagonal matrix (input)
       */
      { evp.solve_small_dense(B_mat, B) } -> std::same_as<std::size_t>;

      { evp.get_current_eigenvalues() };
      { evp.get_current_eigenvectors() } -> std::same_as<const typename EVP::BlockMultivector::BlockMatrix&>;
    } &&
    (
        /** @brief Ensure that the blocksize of the eigenproblem and the blocksize of the BlockMultivector coincide */
        EVP::blocksize == EVP::BlockMultivector::blocksize);

} // namespace trl
