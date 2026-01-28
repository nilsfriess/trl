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
  // , Y(evp->create_blockmatrix(ncv / blocksize, ncv / blocksize))
  // , ritz_values(evp->malloc(ncv))
  {
    // Validate that we won't exhaust the Krylov subspace
    // The maximum number of orthogonal vectors is evp->size()
    // We need ncv + blocksize vectors (ncv blocks plus one trailing block)
    if (ncv + blocksize > evp->size()) {
      throw std::invalid_argument("ncv (" + std::to_string(ncv) + ") + blocksize (" + std::to_string(blocksize) + ") exceeds problem dimension (" + std::to_string(evp->size()) +
                                  "). Krylov subspace would be exhausted. Reduce ncv to at most " + std::to_string(evp->size() - blocksize) + ".");
    }
  }

  // ~BlockLanczos() {
  //   evp->free(ritz_values);
  // }

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

    while (result.iterations < max_restarts) {
      std::cout << "Restart iteration " << result.iterations << " (k=" << k << ", nev/bs=" << nev / blocksize << ", ncv/bs=" << ncv / blocksize << ")\n";
      result.iterations++;

      // Extend the basis to the maximum allowed size
      result.n_op_apply += extend(k, ncv / blocksize);
      // T.print();

      // Solve the small projected system
      evp->solve_small_dense(T);
      const auto& Y = evp->get_current_eigenvectors();
      // const auto& ritz_values = evp->get_current_eigenvalues();

      auto converged = check_convergence();
      // Diagnostic: print residual norms for the current Ritz values
      {
        // std::cout << "Residual norms (first " << nev << "): ";
        // for (unsigned int i = 0; i < nev; ++i) {
        //   std::cout << res_norms[i];
        //   if (i + 1 < nev) std::cout << ", ";
        // }
        // std::cout << "\n";
      }
      if (converged >= nev) {
        result.converged = true;
        // std::cout << "Converged " << converged << " eigenvalues after " << result.iterations << " iterations\n";
        return result;
      }

      // Check if restart is needed (only if nev < ncv)
      if (nev >= ncv) {
        // std::cout << "Maximum basis size reached without convergence (nev >= ncv). Cannot restart.\n";
        result.converged = false;
        return result;
      }

      // Compute Ritz vectors that we will keep
      // Use all Lanczos vectors (0 to ncv/blocksize-1) to compute Ritz vectors,
      // but only keep the first nev/blocksize Ritz vectors in the result
      // Manually compute W = V * Y for only the first nev/blocksize columns of Y
      // std::cout << "  Computing " << nev / blocksize << " Ritz vectors from " << ncv / blocksize << " Lanczos vectors\n";
      // First zero out the blocks we'll write to
      for (std::size_t j = 0; j < nev / blocksize; ++j) {
        W.block_view(j).set_zero();
        // for (std::size_t idx = 0; idx < Wj.rows() * Wj.cols(); ++idx) Wj.data[idx] = 0;
      }
      // Now compute the matrix-vector products
      for (std::size_t j = 0; j < nev / blocksize; ++j) {
        auto Wj = W.block_view(j);
        for (std::size_t i = 0; i < ncv / blocksize; ++i) {
          auto Vi = V.block_view(i);
          auto Yij = Y.block_view(i, j);
          Vi.mult_add(Yij, Wj);
        }
      }
      // Copy V_{m+1} to V_{k+1}
      W.block_view(nev / blocksize).copy_from(V.block_view(ncv / blocksize));

      std::swap(V, W);

      // // Check orthonormality of the restarted basis
      // std::cout << "  Checking orthonormality of first " << nev / blocksize + 1 << " blocks after restart:\n";
      // for (std::size_t i = 0; i <= nev / blocksize; ++i) {
      //   auto Vi = V.block_view(i);
      //   auto check = evp->create_blockmatrix(1, 1);
      //   auto check0 = check.block_view(0, 0);
      //   Vi.dot(Vi, check0);
      //   Scalar max_offdiag = 0;
      //   for (std::size_t r = 0; r < blocksize; ++r) {
      //     for (std::size_t c = 0; c < blocksize; ++c)
      //       if (r != c) max_offdiag = std::max(max_offdiag, std::abs(check0.data[r * blocksize + c]));
      //   }
      //   std::cout << "    Block " << i << ": max off-diag = " << max_offdiag << "\n";
      // }

      converged = check_convergence();
      if (converged >= nev) {
        result.converged = true;
        // std::cout << "Converged " << converged << " eigenvalues after " << result.iterations << " iterations\n";
        return result;
      }

      // Not yet converged, so we need to restart
      // Reset convergence counter for the new basis
      nconv = 0;

#ifndef NDEBUG
      // In debug mode, zero all blocks, also those that we overwrite in the next iteration anyway
      for (std::size_t i = 0; i < ncv / blocksize; ++i) {
        for (std::size_t j = 0; j < ncv / blocksize; ++j) {
          auto Tij = T.block_view(i, j);
          Tij.set_zero();
        }
      }
#endif

      // Put the first nev Ritz values on the diagonal of T
      for (std::size_t i = 0; i < nev / blocksize; ++i) {
        for (std::size_t j = 0; j < nev / blocksize; ++j) {
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
      for (std::size_t i = 0; i < nev / blocksize; ++i) {
        auto Tki = T.block_view(nev / blocksize, i);
        auto Tik = T.block_view(i, nev / blocksize);
        auto Xrow = Y.block_view(ncv / blocksize - 1, i); // use last row of Y from current projected problem

        B.block_view(0, 0).mult(Xrow, Tki);
        Tik.copy_from_transpose(Tki);
      }

      // Now the Lanczos three-term relation is violated, so we do one manual
      // (quasi)-Lanczos step to restore it again. After that, we can proceed
      // using the standard Lanczos algorithm (i.e. call the extend method)
      k = nev / blocksize;

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
        Vj.dot(Vk1, Z0);
        Vk1.subtract_product(Vj, Z0);
      }

      // Step 6: Orthonormalize V_{i+1} to get beta_i (Cholesky factor) and V_{i+1}
      auto beta = B.block_view(0, 0);
      evp->orthonormalize(Vk1, beta);

      // Store beta in the block tridiagonal matrix T
      // beta is upper triangular Cholesky factor: V_old = V_new * beta
      // T should be symmetric, so T[i+1,i] = beta and T[i,i+1] = beta^T
      if (k + 1 < T.block_rows()) {
        auto Ti1_i = T.block_view(k + 1, k);
        Ti1_i.copy_from(beta);

        auto Ti_i1 = T.block_view(k, k + 1);
        Ti_i1.copy_from_transpose(beta);

        // T.print();

        k += 1;
      }
      else {
        // We've exhausted the Krylov subspace (k+1 == ncv/blocksize)
        // We cannot extend further, so we must check convergence and return
        // T.print();
        break;
      }
    }

    // std::cout << "Maximum number of restarts (" << max_restarts << ") reached. Converged " << nconv * blocksize << " out of " << nev << " eigenvalues.\n";
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

  /** @brief Compute residuals and check if the method has converged
   *
   *  Computes residual norms for each Ritz pair and determines convergence.
   *  A Ritz value is considered converged if its residual norm is below
   *  a threshold based on the tolerance and machine epsilon.
   *
   *  @returns The number of converged eigenvalues
   *
   *  @note TODO (GPU optimization): For GPU backends, this method requires
   *  host synchronization to compare residual norms against thresholds.
   *  To avoid this, implement a device-side convergence check kernel that:
   *  1. Computes residual norms on device
   *  2. Compares against thresholds on device
   *  3. Returns only a single boolean/count via a reduction
   *  See Ginkgo's stopping criterion implementation for reference:
   *  they use device_storage arrays and run the convergence check entirely
   *  on device, only syncing the final all_converged/one_changed booleans.
   */
  unsigned int check_convergence()
  {
    // Compute residuals: ||r_i|| = ||beta * e_m^T * y_i||
    // where y_i is the i-th eigenvector of T (i-th column of Y)
    // and e_m^T is the last row of the identity (selects last blocksize rows of Y)

    compute_residual_norms();

    // Machine epsilon scaled threshold
    const Scalar eps23 = std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2) / Scalar(3));

    // Count converged blocks (not individual eigenvalues)
    unsigned int nconv_blocks = 0;
    const unsigned int max_blocks = nev / blocksize;

    for (unsigned int bidx = nconv; bidx < max_blocks; ++bidx) {
      bool block_converged = true;

      // Check all eigenvalues in this block
      for (unsigned int j = 0; j < blocksize; ++j) {
        const unsigned int idx = bidx * blocksize + j;

        // Specific threshold for this eigenvalue
        // Use max of relative tolerance and scaled epsilon

        // TODO: Here we access ritz_values directly, but they live on device, not on host!
        const Scalar thresh = std::max(tolerance // * std::abs(ritz_values[idx])
                                       ,
                                       eps23);

        if (res_norms[idx] > thresh) {
          block_converged = false;
          break;
        }
      }

      if (block_converged) nconv_blocks++;
      else break; // Stop at first unconverged block
    }

    nconv += nconv_blocks;
    return nconv * blocksize;
  }

private:
  /** @brief Compute residual norms for all Ritz pairs
   *
   *  Computes ||r_i|| = ||beta * e_m^T * y_i|| for each eigenvector y_i.
   *  The residual is the norm of beta (last off-diagonal block) times
   *  the last blocksize elements of the i-th eigenvector.
   *
   *  @note TODO (GPU optimization): two_norm_on_host() requires a device-to-host
   *  sync for GPU backends. Consider fusing the norm computation and convergence
   *  check into a single device kernel that returns only a count of converged
   *  eigenvalues, avoiding the transfer of individual residual norms.
   */
  // void compute_residual_norms()
  // {
  //   // beta is the last off-diagonal block of T
  //   auto beta = B.block_view(0, 0); // We stored it in B during extend()

  //   std::cout << "  Computing residuals: beta norm = " << std::sqrt(beta.data[0] * beta.data[0] + beta.data[1] * beta.data[1] + beta.data[2] * beta.data[2] + beta.data[3] * beta.data[3]) << "\n";

  //   // For each eigenvalue we want to check
  //   for (unsigned int i = nconv * blocksize; i < nev; ++i) {
  //     // Extract the last blocksize elements of eigenvector i from Y
  //     // Y is stored as a BlockMatrix with block_rows() x block_cols() blocks
  //     // We need the elements from the last block row for column i

  //     std::vector<Scalar> last_elements(blocksize);
  //     std::vector<Scalar> result(blocksize);

  //     // Get the last block row index
  //     const auto& Y = evp->get_current_eigenvectors();
  //     const unsigned int last_block_row = Y.block_rows() - 1;
  //     const unsigned int col_block = i / blocksize;

  //     // Extract last blocksize elements from eigenvector i
  //     // This is a bit tricky because Y is stored in blocks
  //     auto Y_block = Y.block_view(last_block_row, col_block);
  //     const unsigned int col_in_block = i % blocksize;

  //     for (unsigned int j = 0; j < blocksize; ++j) last_elements[j] = Y_block(j, col_in_block);

  //     // Compute beta * last_elements
  //     beta.mult(last_elements, result);

  //     // Compute norm of result
  //     Scalar sum_sq = 0;
  //     for (unsigned int j = 0; j < blocksize; ++j) sum_sq += result[j] * result[j];
  //     res_norms[i] = std::sqrt(sum_sq);

  //     if (i == 0) {
  //       std::cout << "    Eigenvalue 0: last_Y = [";
  //       for (unsigned int j = 0; j < blocksize; ++j) {
  //         std::cout << last_elements[j];
  //         if (j + 1 < blocksize) std::cout << ", ";
  //       }
  //       std::cout << "], residual = " << res_norms[i] << "\n";
  //     }
  //   }
  // }
  void compute_residual_norms()
  {
    auto beta = B.block_view(0, 0);
    auto result = B.block_view(0, 1);
    const auto& Y = evp->get_current_eigenvectors();

    for (std::size_t block = 0; block < nev / blocksize; ++block) {
      auto Y_block = Y.block_view(Y.block_rows() - 1, block);
      beta.mult(Y_block, result);

      const auto& residuals = evp->two_norm_on_host(result);
      for (std::size_t j = 0, i = block * blocksize; i < (block + 1) * blocksize; ++i, ++j) res_norms[i] = residuals[j];
    }

    // for (unsigned int i = nconv * blocksize; i < nev; ++i) {
    //   const unsigned int last_block_row = Y.block_rows() - 1;
    //   const unsigned int col_block = i / blocksize;
    //   const unsigned int col_in_block = i % blocksize;

    //   auto Y_block = Y.block_view(last_block_row, col_block);

    //   // Extract column (stays on device for GPU backend)
    //   Y_block.get_column(col_in_block, last_elements);

    //   // Multiply (on device for GPU backend)
    //   beta.mult(last_elements, result);

    //   // Compute norm (on device for GPU backend, returns scalar to host)
    //   res_norms[i] = result.norm();

    //   // Diagnostic (only transfer data when actually printing)
    //   if (i == 0) {
    //     auto host_copy = last_elements.to_host();
    //     std::cout << "    Eigenvalue 0: last_Y = [";
    //     for (unsigned int j = 0; j < blocksize; ++j) {
    //       std::cout << host_copy[j];
    //       if (j + 1 < blocksize) std::cout << ", ";
    //     }
    //     std::cout << "], residual = " << res_norms[i] << "\n";
    //   }
    // }
  }

public:
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

      // Step 5: Full reorthogonalization (uses same inner product as evp->dot)
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
