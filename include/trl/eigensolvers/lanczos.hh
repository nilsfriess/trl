#pragma once

#include "../concepts.hh"
#include "trl/eigensolvers/params.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

namespace trl {
template <Eigenproblem EVP>
class BlockLanczos {
public:
  using BMV = typename EVP::BlockMultivector;
  using Scalar = typename BMV::Scalar;
  static constexpr unsigned int blocksize = BMV::blocksize;

  BlockLanczos(std::shared_ptr<EVP> evp_, const EigensolverParams& params)
      : evp(std::move(evp_))
      , nev(params.nev)
      , ncv(params.ncv)
      , max_restarts(params.max_restarts)
      , res_norms(nev, 0)
      , V(evp->create_multivector(evp->size(), ncv + EVP::blocksize))
      , W(evp->create_multivector(evp->size(), ncv + EVP::blocksize))
      // Allocate W with ncv columns to match matrix dimensions
      , T(evp->create_blockmatrix(ncv / blocksize, ncv / blocksize))
      , B(evp->create_blockmatrix(1, 2))
  {
    // Validate that we won't exhaust the Krylov subspace
    // The maximum number of orthogonal vectors is evp->size()
    // We need ncv + blocksize vectors (ncv blocks plus one trailing block)
    if (ncv + blocksize > evp->size()) {
      throw std::invalid_argument("ncv (" + std::to_string(ncv) + ") + blocksize (" + std::to_string(blocksize) + ") exceeds problem dimension (" + std::to_string(evp->size()) +
                                  "). Krylov subspace would be exhausted. Reduce ncv to at most " + std::to_string(evp->size() - blocksize) + ".");
    }

    if (nev >= ncv) throw std::invalid_argument("nev must be strictly smaller than ncv");

    // Validate that we have enough space for thick restart
    // After restart, we keep k_restart = nev/blocksize + 1 blocks, and need at least 1 more to extend
    unsigned int min_ncv_blocks = nev / blocksize + 2;
    if (ncv / blocksize < min_ncv_blocks) {
      throw std::invalid_argument("ncv (" + std::to_string(ncv) + ") is too small for thick restart with nev=" + std::to_string(nev) + " and blocksize=" + std::to_string(blocksize) +
                                  ". Minimum required: ncv >= " + std::to_string(min_ncv_blocks * blocksize) + ".");
    }

    if constexpr (requires { evp->set_tolerance(Scalar(0)); }) evp->set_tolerance(static_cast<Scalar>(params.tolerance));
  }

  /** @brief Solves the eigenvalue problem using thick-restart Lanczos
   */
  EigensolverResult solve()
  {
    EigensolverResult result{
        .converged = false,
        .iterations = 0,
        .n_op_apply = 0,
    };
    unsigned int k = 0;

    auto beta = B.block_view(0, 0);

    while (result.iterations < max_restarts) {
      // std::cout << "Restart iteration " << result.iterations << " (k=" << k << ", nev/bs=" << nev / blocksize << ", ncv/bs=" << ncv / blocksize << ")\n";
      result.iterations++;

      // Extend the basis to the maximum allowed size. In the first iteration, k = 0 so here
      // the initial Lanczos decomposition is built. In subsequent iterations, k = nev / blocksize
      // because we retain the first nev blocks as restart vectors.
      result.n_op_apply += extend(k, ncv / blocksize);

      // Solve the small projected system
      auto converged = evp->solve_small_dense(T, beta, nev);
      if (converged >= nev) {
        result.converged = true;
        return result;
      }
      else {
        // std::cout << "  Eigenvalues converged: " << converged << "\n";
      }

      // We did not converge yet, so now we must prepare for restart. We begin by computing
      // the Ritz vectors that we will keep.
      auto k_restart = nev / blocksize + 1;

      // Check if we have enough space for restart
      if (k_restart >= ncv / blocksize) {
        // Not enough space to restart - return with partial convergence
        break;
      }

      const auto& Y = evp->get_current_eigenvectors();
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

      // Reset convergence counter for the new basis
      nconv = 0;
      // Put the first nev Ritz values on the diagonal of T
      for (std::size_t i = 0; i < k_restart; ++i) {
        for (std::size_t j = 0; j < k_restart; ++j) {
          auto Tij = T.block_view(i, j);
          Tij.set_zero();

          if (i == j) {
            const auto& evals = evp->get_eigenvalues_block(i);
            // std::cout << "DEBUG restart: Setting T(" << i << "," << i << ") to eigenvalue " << evals.get_const_data()[0] << "\n";
            Tij.set_diagonal(evals);
          }
        }
      }
      // Put the "residual block" into T
      for (std::size_t i = 0; i < k_restart; ++i) {
        auto Tki = T.block_view(k_restart, i);
        auto Tik = T.block_view(i, k_restart);
        auto Xrow = Y.block_view(ncv / blocksize - 1, i); // use last row of Y from current projected problem

        B.block_view(0, 0).mult(Xrow, Tki);
        Tik.copy_from_transpose(Tki);
      }

      // Now the Lanczos three-term relation is violated, so we do one manual
      // (quasi)-Lanczos step to restore it again. After that, we can proceed
      // using the standard Lanczos algorithm (i.e. call the extend method)
      k = k_restart;

      // Apply the operator
      auto Vk = V.block_view(k);
      auto Vk1 = V.block_view(k + 1);
      evp->apply(Vk, Vk1);
      result.n_op_apply++;

      // Compute the next diagonal block
      auto Tkk = T.block_view(k, k);
      evp->dot(Vk, Vk1, Tkk);

      auto W0 = W.block_view(0);    // temp storage
      auto Z0 = B.block_view(0, 1); // temp storage
      Vk.mult(Tkk, W0);
      Vk1 -= W0;

      // Subtract the contribution from V_{k-1} * beta_{k-1}^T
      if (k > 0) {
        auto Vk_prev = V.block_view(k - 1);
        auto beta_prev = T.block_view(k, k - 1);
        Vk_prev.mult_transpose(beta_prev, W0);
        Vk1 -= W0;
      }

      // Reorthogonalise against all previous blocks
      for (unsigned int j = 0; j < k + 1; ++j) {
        auto Vj = V.block_view(j);
        evp->dot(Vj, Vk1, Z0);
        Vk1.subtract_product(Vj, Z0);
      }

      // Step 6: Orthonormalize V_{i+1} to get beta_i (Cholesky factor) and V_{i+1}
      evp->orthonormalize(Vk1, beta);

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
        // We've exhausted the Krylov subspace (k+1 == ncv/blocksize)
        // We cannot extend further, so let's just return.
        break;
      }
    }

    return result;
  }

  /** @brief Get the initial block V_0 for initialization
   *
   *  The user must initialize this block before calling extend(0, m).
   */
  auto initial_block() { return V.block_view(0); }

  /** @brief Return the current Lanczos vectors */
  auto& get_basis() { return V; }

  /** @brief Return the block tridiagonal matrix T */
  auto& get_T() { return T; }

  /** @brief Return the B matrix containing beta values */
  auto& get_B() { return B; }

  /** @brief Return the number of converged eigenvalues */
  unsigned int get_nconv() const { return nconv * blocksize; }

  // const auto* get_eigenvalues() const { return ritz_values; }

  /** @brief Extend the Lanczos basis from a step-k block factorisation to a step-m block factorisation
   *
   *  @note The parameters k and m are counted in blocks.
   *
   *  @returns The number of operator applications (i.e. the number of calls to evp->apply)
   */
  unsigned int extend(unsigned int k, unsigned int m)
  {
    assert(k < m);
    assert(m <= ncv / blocksize);

    unsigned int n_op_apply = 0;

    auto W0 = W.block_view(0);
    auto beta = B.block_view(0, 0);
    auto Z0 = B.block_view(0, 1); // temp storage

    // Orthonormalize the initial block if starting from k=0
    if (k == 0) {
      auto V0 = V.block_view(0);
      evp->orthonormalize(V0, Z0); // Use Z0 as temp storage for R matrix
    }

    for (unsigned int i = k; i < m; ++i) {
      auto V_curr = V.block_view(i);
      auto V_next = V.block_view(i + 1);

      // Step 1: v_{i+1} = A v_i
      evp->apply(V_curr, V_next);
      n_op_apply++;

      // Step 2: v_{i+1} -= v_{i-1} * beta_{i-1}^T
      if (i > 0) {
        auto V_prev = V.block_view(i - 1);
        V_prev.mult_transpose(beta, W0);
        V_next -= W0;
      }

      // Step 3: Compute T(i,i) = <v_i, v_{i+1}>
      auto Tii = T.block_view(i, i);
      evp->dot(V_curr, V_next, Tii);

      // Step 4: Orthogonalise v_{i+1} -= v_i * T(i,i)
      V_curr.mult(Tii, W0);
      V_next -= W0;

      // Step 5: Full reorthogonalization
      for (unsigned int j = 0; j < i + 1; ++j) {
        auto Vj = V.block_view(j);

        evp->dot(Vj, V_next, Z0);
        V_next.subtract_product(Vj, Z0);
      }

      // Step 6: Orthonormalize V_{i+1} to get beta_i (Cholesky factor) and V_{i+1}
      evp->orthonormalize(V_next, beta);

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

private:
  std::shared_ptr<EVP> evp;

  // Parameters
  const unsigned int nev;
  const unsigned int ncv;
  const unsigned int max_restarts;
  const Scalar tolerance = 1e-8; // Default tolerance for convergence

  // Convergence tracking
  unsigned int nconv = 0;        // Number of converged blocks
  std::vector<Scalar> res_norms; // Residual norms for each eigenvalue

  // Vectors and matrices
  BMV V;
  BMV W; // Temp vector

  typename BMV::BlockMatrix T; // Block tridiagonal matrix
  typename BMV::BlockMatrix B; // Temp matrix
};
} // namespace trl
