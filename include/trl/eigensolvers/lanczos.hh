#pragma once

#include "../concepts.hh"
#include "params.hh"
#include "reorthogonalization.hh"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <numeric>

namespace trl {
/** @brief Block Lanczos eigensolver with thick restart.
 *
 *  Computes a subset of eigenvalues and eigenvectors of the eigenproblem
 *  defined by @p EVP using a restarted block Krylov iteration.
 *
 *  @tparam EVP  Eigenproblem type satisfying the \ref trl::Eigenproblem concept.
 *  @tparam Reorth  Reorthogonalization strategy. Must satisfy
 *          \ref trl::ReorthogonalizationStrategy. Defaults to \ref trl::ModifiedGS.
 */
template <BackendConcept B, OperatorConcept<B> O, class Reorth = ModifiedGS>
// requires ReorthogonalizationStrategy<Reorth, EVP, typename EVP::BlockMultivector>
class BlockLanczos {
public:
  using BMV = typename B::Multivector;
  using Scalar = typename B::Scalar;
  static constexpr unsigned int blocksize = B::blocksize;

  BlockLanczos(B backend_, std::shared_ptr<O> op_, const EigensolverParams& params)
      : nev(params.nev)
      , ncv(params.ncv)
      , max_restarts(params.max_restarts)
      , tolerance(params.tolerance)
      , V(backend_.make_multivector(op_->size(), ncv + blocksize))
      , W(backend_.make_multivector(op_->size(), ncv + blocksize))
      , // Allocate W with ncv columns to match matrix dimensions
      T(backend_.make_blockmatrix(ncv / blocksize, ncv / blocksize))
      , U(backend_.make_blockmatrix(1, 2))
      , Y(backend_.make_blockmatrix(ncv / blocksize, ncv / blocksize))
      , op(std::move(op_))
      , backend(std::move(backend_))
  {
    if (nev % blocksize != 0) throw std::invalid_argument("nev (" + std::to_string(nev) + ") must be a multiple of blocksize (" + std::to_string(blocksize) + ").");

    // Validate that we won't exhaust the Krylov subspace
    // The maximum number of orthogonal vectors is op->size()
    // We need ncv + blocksize vectors (ncv blocks plus one trailing block)
    if (ncv + blocksize > op->size()) {
      throw std::invalid_argument("ncv (" + std::to_string(ncv) + ") + blocksize (" + std::to_string(blocksize) + ") exceeds problem dimension (" + std::to_string(op->size()) +
                                  "). Krylov subspace would be exhausted. Reduce ncv to at most " + std::to_string(op->size() - blocksize) + ".");
    }

    if (nev >= ncv) throw std::invalid_argument("nev must be strictly smaller than ncv");

    // After restart we keep nev/blocksize Ritz blocks + 1 residual block, and need
    // room for at least 1 more block to extend before the next solve
    unsigned int min_ncv_blocks = nev / blocksize + 2;
    if (ncv / blocksize < min_ncv_blocks) {
      throw std::invalid_argument("ncv (" + std::to_string(ncv) + ") is too small for thick restart with nev=" + std::to_string(nev) + " and blocksize=" + std::to_string(blocksize) +
                                  ". Minimum required: ncv >= " + std::to_string(min_ncv_blocks * blocksize) + ".");
    }
  }

  /** @brief Solves the eigenvalue problem using thick-restart Lanczos
   */
  EigensolverResult<Scalar> solve()
  {
    EigensolverResult<Scalar> result{
        .converged = false,
        .iterations = 0,
        .n_op_apply = 0,
        .eigenvalues = {},
    };
    unsigned int k = 0;

    auto beta = U.block_view(0, 0);

    while (result.iterations < max_restarts) {
      result.iterations++;

      // Extend the basis up to the ncv budget: a full build when k == 0, a continuation after restart otherwise.
      result.n_op_apply += extend(k, ncv / blocksize);

      // Solve the small projected system
      auto converged = solve_small_dense(beta);
      if (converged >= nev) {
        result.converged = true;
        result.eigenvalues = std::move(eigenvalues);
        return result;
      }

      // We did not converge yet, so now we must prepare for restart. We begin by computing
      // the Ritz vectors that we will keep.
      auto k_restart = nev / blocksize;
      assert(k_restart < ncv / blocksize);

      for (std::size_t j = 0; j < k_restart; ++j) {
        auto Wj = W.block_view(j);
        Wj.set_zero();
        for (std::size_t i = 0; i < ncv / blocksize; ++i) {
          auto Vi = V.block_view(i);
          auto Yij = Y.block_view(i, j);
          Vi.mult_add(Yij, Wj);
        }
      }
      // Copy V_{m+1} to V_{k+1}
      W.block_view(k_restart).copy_from(V.block_view(ncv / blocksize));

      std::swap(V, W);

      // Put the first nev Ritz values on the diagonal of T
      for (std::size_t i = 0; i < k_restart; ++i) {
        for (std::size_t j = 0; j < k_restart; ++j) {
          auto Tij = T.block_view(i, j);
          Tij.set_zero();

          if (i == j) {
            const auto& evals = get_eigenvalues_block(i);
            Tij.set_diagonal(evals);
          }
        }
      }
      // Put the "residual block" into T
      for (std::size_t i = 0; i < k_restart; ++i) {
        auto Tki = T.block_view(k_restart, i);
        auto Tik = T.block_view(i, k_restart);
        auto Xrow = Y.block_view(ncv / blocksize - 1, i); // use last row of Y from current projected problem

        U.block_view(0, 0).mult(Xrow, Tki);
        Tik.copy_from_transpose(Tki);
      }

      // Now the Lanczos three-term relation is violated, so we do one manual
      // (quasi)-Lanczos step to restore it again. After that, we can proceed
      // using the standard Lanczos algorithm (i.e. call the extend method)
      k = k_restart;

      // Apply the operator
      auto Vk = V.block_view(k);
      auto Vk1 = V.block_view(k + 1);
      op->apply(Vk, Vk1);
      result.n_op_apply++;

      // Compute the next diagonal block
      auto Tkk = T.block_view(k, k);
      op->dot(Vk, Vk1, Tkk);

      auto W0 = W.block_view(0);    // temp storage
      auto Z0 = U.block_view(0, 1); // temp storage
      Vk.mult(Tkk, W0);
      Vk1 -= W0;

      // Vk1 couples to every retained Ritz block here, not just Vk-1, so skip straight
      // to full reorthogonalization instead of a single-neighbor subtraction.
      reorthogonalize_against(Vk1, k + 1, Z0);

      // Step 6: Orthonormalize V_{i+1} to get beta_i (Cholesky factor) and V_{i+1}
      orthonormalize(Vk1, beta);

      // Store beta in the block tridiagonal matrix T
      // beta is upper triangular Cholesky factor: V_old = V_new * beta
      // T should be symmetric, so T[i+1,i] = beta and T[i,i+1] = beta^T
      if (k + 1 < T.block_rows()) {
        auto Ti1_i = T.block_view(k + 1, k);
        Ti1_i.copy_from(beta);

        auto Ti_i1 = T.block_view(k, k + 1);
        Ti_i1.copy_from_transpose(beta);

        k += 1;
      }
      else {
        TRL_TODO("Should this ever happen?");
        // We've exhausted the Krylov subspace (k+1 == ncv/blocksize)
        // We cannot extend further, so let's just return.
        break;
      }
    }

    // Report the best Ritz values we have even when we ran out of restarts, so
    // callers can always read result.eigenvalues without checking `converged`.
    result.eigenvalues = std::move(eigenvalues);
    return result;
  }

  /** @brief Get the initial block V_0 for initialization
   *
   *  The user must initialize this block before calling solve().
   */
  auto initial_block() { return V.block_view(0); }

  /** @brief Extend the Lanczos basis from a step-k block factorisation to a step-m block factorisation
   *
   *  @note The parameters k and m are counted in blocks.
   *
   *  @returns The number of operator applications (i.e. the number of calls to backend.apply)
   */
  unsigned int extend(unsigned int k, unsigned int m)
  {
    assert(k < m);
    assert(m <= ncv / blocksize);

    unsigned int n_op_apply = 0;

    auto W0 = W.block_view(0);
    auto beta = U.block_view(0, 0);
    auto Z0 = U.block_view(0, 1); // temp storage

    // Orthonormalize the initial block if starting from k=0
    if (k == 0) {
      auto V0 = V.block_view(0);
      orthonormalize(V0, Z0); // Use Z0 as temp storage for R matrix
    }

    for (unsigned int i = k; i < m; ++i) {
      auto V_curr = V.block_view(i);
      auto V_next = V.block_view(i + 1);

      // Step 1: v_{i+1} = A v_i
      op->apply(V_curr, V_next);
      n_op_apply++;

      // Step 2: v_{i+1} -= v_{i-1} * beta_{i-1}^T
      if (i > 0) {
        auto V_prev = V.block_view(i - 1);
        V_prev.mult_transpose(beta, W0);
        V_next -= W0;
      }

      // Step 3: Compute T(i,i) = <v_i, v_{i+1}>
      auto Tii = T.block_view(i, i);
      op->dot(V_curr, V_next, Tii);

      // Step 4: Orthogonalise v_{i+1} -= v_i * T(i,i)
      V_curr.mult(Tii, W0);
      V_next -= W0;

      // Step 5: Full reorthogonalization
      reorthogonalize_against(V_next, i + 1, Z0);

      // Step 6: Orthonormalize V_{i+1} to get beta_i (Cholesky factor) and V_{i+1}
      orthonormalize(V_next, beta);

      // Store beta in the block tridiagonal matrix T
      // beta is upper triangular Cholesky factor: V_old = V_new * beta
      // The Lanczos relation is: A*V_i = ... + V_{i+1} * beta_i
      // So T[i+1,i] = beta and T[i,i+1] = beta^T for symmetry
      if (i + 1 < T.block_rows()) {
        auto Ti1_i = T.block_view(i + 1, i);
        Ti1_i.copy_from(beta);

        auto Ti_i1 = T.block_view(i, i + 1);
        Ti_i1.copy_from_transpose(beta);
      }
    }

    return n_op_apply;
  }

  /** @brief Return the current Lanczos vectors */
  auto& get_basis() { return V; }

  /** @brief Return the block tridiagonal matrix T */
  auto& get_T() { return T; }

  /** @brief Return the B matrix containing beta values */
  auto& get_beta() { return U; }

private:
  unsigned int solve_small_dense(typename B::BlockMatrix::BlockView beta)
  {
    const auto n_total = T.block_rows() * blocksize;

    // Convert block matrix to dense Eigen matrix
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> B_dense(n_total, n_total);

    for (std::size_t i = 0; i < T.block_rows(); ++i) {
      for (std::size_t j = 0; j < T.block_cols(); ++j) {
        auto host_block = backend.host_block(T.block_view(i, j), Access::Read);
        for (unsigned int bi = 0; bi < blocksize; ++bi)
          for (unsigned int bj = 0; bj < blocksize; ++bj) B_dense(i * blocksize + bi, j * blocksize + bj) = host_block[bi * blocksize + bj];
      }
    }

    // Compute eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> solver(B_dense);
    if (solver.info() != Eigen::Success) throw std::runtime_error("Eigendecomposition failed");

    // Store eigenvalues in descending order (largest first)
    // Eigen returns them in ascending order, so reverse
    std::vector<unsigned int> indices(n_total);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](const auto& i, const auto& j) { return std::abs(solver.eigenvalues()[i]) > std::abs(solver.eigenvalues()[j]); });

    eigenvalues.resize(n_total);
    for (std::size_t i = 0; i < n_total; ++i) eigenvalues[i] = solver.eigenvalues()(indices[i]);

    // Store eigenvectors in BlockMatrix format (also reversed to match eigenvalues)
    for (std::size_t i = 0; i < T.block_rows(); ++i) {
      for (std::size_t j = 0; j < T.block_cols(); ++j) {
        auto host_block = backend.host_block(Y.block_view(i, j), Access::Write);
        for (unsigned int bi = 0; bi < blocksize; ++bi) {
          for (unsigned int bj = 0; bj < blocksize; ++bj) {
            // Reverse column order to match descending eigenvalue order
            host_block[bi * blocksize + bj] = solver.eigenvectors()(i * blocksize + bi, indices[j * blocksize + bj]);
          }
        }
      }
    }

    // Compute residual norms and count converged eigenvalues
    // Residual norm: ||beta * v_j||_2 where v_j are the last blocksize components of eigenvector j
    std::size_t n_converged = 0;
    const std::size_t last_block_row = Y.block_rows() - 1;
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();

    auto beta_host = backend.host_block(beta, Access::Read);
    const std::size_t n_check = std::min<std::size_t>(nev, n_total);
    for (std::size_t col_idx = 0; col_idx < n_check; ++col_idx) {
      const std::size_t block_col = col_idx / blocksize;
      const std::size_t col_in_block = col_idx % blocksize;

      auto v_last = Y.block_view(last_block_row, block_col);
      auto v_last_host = backend.host_block(v_last, Access::Read);

      // Compute beta * v_j (where v_j is a column vector of size blocksize)
      Scalar norm_sq = 0.0;
      for (unsigned int i = 0; i < blocksize; ++i) {
        Scalar sum = 0.0;
        for (unsigned int k = 0; k < blocksize; ++k) sum += beta_host[i * blocksize + k] * v_last_host[k * blocksize + col_in_block];
        norm_sq += sum * sum;
      }

      Scalar residual_norm = std::sqrt(norm_sq);
      const Scalar theta = eigenvalues[col_idx];
      const Scalar denom = std::max(std::abs(theta), eps);
      const Scalar rel_residual = residual_norm / denom;

      if (rel_residual < tolerance) n_converged++;
    }

    return n_converged;
  }

  std::span<Scalar, blocksize> get_eigenvalues_block(unsigned int block)
  {
    std::span<Scalar, blocksize> ev_block(eigenvalues.data() + block * blocksize, blocksize);
    return ev_block;
  }

  void orthonormalize(typename BMV::BlockView V_next, typename B::BlockMatrix::BlockView beta)
  {
    // 1. Compute Gram matrix G = V^T * V (stored in R).
    // Use the operator's inner product (which is the B-inner product for
    // generalized problems), not the Euclidean dot of the view.
    op->dot(V_next, V_next, beta);

    // 2. Compute Cholesky factorization of G = U^T * U
    //
    // The host mirror only reaches the device when it is destroyed, so each
    // scope below ends where the device-side value of beta has to be current.
    Eigen::Matrix<Scalar, blocksize, blocksize> stored_R;
    {
      auto beta_host = backend.host_block(beta, Access::ReadWrite);
      Eigen::Map<Eigen::Matrix<Scalar, blocksize, blocksize, Eigen::RowMajor>> RR(beta_host.data());
      Eigen::LLT<Eigen::Matrix<Scalar, blocksize, blocksize>> llt(RR);
      if (llt.info() != Eigen::Success) throw std::runtime_error("Cholesky factorization failed in orthonormalize");
      RR = llt.matrixL().transpose();
      stored_R = RR;

      // 3. Compute U^{-1} and store in R temporarily
      RR = stored_R.inverse().eval();
    } // flush: beta holds U^{-1} on the device

    auto Vtemp0 = W.block_view(0);
    V_next.mult(beta, Vtemp0); // V_temp = V * U^{-1}
    V_next.copy_from(Vtemp0);

    // 4. Restore U in R (the Cholesky factor, not its inverse)
    {
      auto beta_host = backend.host_block(beta, Access::Write);
      Eigen::Map<Eigen::Matrix<Scalar, blocksize, blocksize, Eigen::RowMajor>>(beta_host.data()) = stored_R;
    } // flush: beta holds U on the device
  }

  // Parameters
  unsigned int nev;
  unsigned int ncv;
  unsigned int max_restarts;
  double tolerance;

  std::vector<Scalar> eigenvalues;

  // Vectors and matrices
  BMV V;
  BMV W; // Temp vector

  typename B::BlockMatrix T; // Block tridiagonal matrix
  typename B::BlockMatrix U; // Temp matrix
  typename B::BlockMatrix Y; // Eigenvectors of small problem

  Reorth reorth_{};

  void reorthogonalize_against(typename BMV::BlockView V_next, unsigned int count, typename B::BlockMatrix::BlockView tmp) { reorth_(*op, V, count, V_next, tmp); }

  std::shared_ptr<O> op;
  [[no_unique_address]] B backend;
};
} // namespace trl
