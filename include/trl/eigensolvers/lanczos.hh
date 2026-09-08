#pragma once

#include "../concepts.hh"
#include "../helpers.hh"
#include "params.hh"
#include "reorthogonalization.hh"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace trl {
/** @brief Block Lanczos eigensolver with thick restart.
 *
 *  Computes a subset of eigenvalues and eigenvectors of the eigenproblem
 *  defined by @p O using a restarted block Lanczos iteration with full
 *  reorthogonalization.
 *
 *  In fact, we're actually doing an Arnoldi iteration here, i.e. we never
 *  rely on symmetry of the problem and the Lanczos three-term recurrence
 *  (at least not in the Krylov basis extension part of the algorithm; the
 *  small projected eigenproblem does assume symmetry for now).
 *
 *  One step of the algorithm is roughly:
 *  - apply the operator,
 *  - project the result against the whole basis
 *  - orthonormalise using full reorthogonalisation
 *
 *
 *  @tparam B  Backend type satisfying \ref trl::BackendConcept.
 *  @tparam O  Operator type satisfying \ref trl::OperatorConcept.
 *  @tparam Reorth  Reorthogonalization strategy, defaulting to
 *          \ref trl::ClassicalGS2. It must leave the total projection in the
 *          coefficient view it is handed, since that view is a column of T.
 */
template <BackendConcept B, OperatorConcept<B> O, class Reorth = ClassicalGS2>
class BlockLanczos {
public:
  using BMV = typename B::Multivector;
  using Scalar = typename B::Scalar;
  using DenseMatrix = typename B::DenseMatrix;
  using Panel = typename BMV::PanelView;
  static constexpr unsigned int blocksize = B::blocksize;

  static_assert(ReorthogonalizationStrategy<Reorth, O, B, Panel, DenseMatrix>, "The reorthogonalization strategy must be callable as (op, backend, panel, block, coefficients, scratch)");

  BlockLanczos(B backend_, std::shared_ptr<O> op_, const EigensolverParams& params)
      : nev(params.nev)
      , ncv(params.ncv)
      , max_restarts(params.max_restarts)
      , tolerance(params.tolerance)
      , V(backend_.make_multivector(op_->size(), ncv + blocksize))
      , W(backend_.make_multivector(op_->size(), ncv + blocksize))
      , T_(backend_.make_dense_matrix(ncv, ncv))
      , YY_(backend_.make_dense_matrix(ncv, nev))
      , H2_(backend_.make_dense_matrix(ncv + blocksize, blocksize))
      , gram_(backend_.make_dense_matrix(blocksize, blocksize))
      , beta_inv_(backend_.make_dense_matrix(blocksize, blocksize))
      // beta and the Cholesky status share one allocation so that the restart
      // reads both with a single transfer.
      , bstat_(backend_.make_dense_matrix(blocksize + 1, blocksize))
      , op(std::move(op_))
      , backend(std::move(backend_))
  {
    if (nev % blocksize != 0) throw std::invalid_argument("nev (" + std::to_string(nev) + ") must be a multiple of blocksize (" + std::to_string(blocksize) + ").");
    if (ncv % blocksize != 0) throw std::invalid_argument("ncv (" + std::to_string(ncv) + ") must be a multiple of blocksize (" + std::to_string(blocksize) + ").");

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

  /** @brief Solves the eigenvalue problem using thick-restart Lanczos */
  EigensolverResult<Scalar> solve()
  {
    EigensolverResult<Scalar> result{
        .converged = false,
        .iterations = 0,
        .n_op_apply = 0,
        .eigenvalues = {},
    };

    const unsigned int m = ncv / blocksize;
    const unsigned int k_restart = nev / blocksize;
    unsigned int k = 0;

    while (result.iterations < max_restarts) {
      result.iterations++;

      // Extend the basis up to the ncv size (for k=0 this is the initial full build of the basis,
      // when k != 0, this extends the basis again to the full size after a thick restart.
      result.n_op_apply += extend(k, m);

      // Solve the small projected system. When it does not converge this also
      // rebuilds T and the Ritz coefficients for the restart below.
      const auto converged = solve_small_dense();
      if (converged >= nev) {
        result.converged = true;
        result.eigenvalues = std::move(eigenvalues);
        return result;
      }

      // Thick restart: keep the nev Ritz vectors and the trailing residual
      // block. W(:, 0:nev) = V(:, 0:ncv) * YY, one panel product.
      V.panel_view(0, m).mult(TransposeMode::NoTranspose, YY_, W.panel_view(0, k_restart));
      W.panel_view(k_restart, 1).copy_from(V.panel_view(m, 1));
      std::swap(V, W);

      // T restarts as the kept Ritz values on the diagonal, written by
      // prepare_restart. The "arrowhead" that couples them to the residual block
      // is column k_restart of T, which is computed by the extend method above.
      k = k_restart;
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
   *  @returns The number of operator applications (i.e. the number of calls to op->apply)
   */
  unsigned int extend(unsigned int k, unsigned int m)
  {
    assert(k < m);
    assert(m <= ncv / blocksize);

    unsigned int n_op_apply = 0;

    // Cleared once per sweep and read back at the next restart, so that a
    // failed factorisation costs no synchronisation until it is reported.
    status_view().fill_zero();

    if (k == 0) orthonormalize(V.panel_view(0, 1));

    for (unsigned int i = k; i < m; ++i) {
      auto Vp = V.panel_view(0, i + 1);
      auto w = V.panel_view(i + 1, 1);

      op->apply(V.panel_view(i, 1), w);
      n_op_apply++;

      // h is column block i of T: the projection coefficients the
      // orthogonalization needs are exactly that column of V^T A V.
      auto h = T_.view().block(0, i * blocksize, (i + 1) * blocksize, blocksize);
      reorth_(*op, backend, Vp, w, h, H2_.view().block(0, 0, (i + 1) * blocksize, blocksize));

      // Here we set T(i-1, i) = beta_{i-1}^T, where beta comes from the previous step's Cholesky
      // factorisation. This is more accurate than using the recomputed value that's currently
      // in T (TODO: Reference).
      if (i > k) T_.view().block((i - 1) * blocksize, i * blocksize, blocksize, blocksize).copy_from_transpose(beta());

      orthonormalize(w);
    }

    return n_op_apply;
  }

  /** @brief Return the current Lanczos vectors */
  auto& get_basis() { return V; }

  /** @brief Return the projected matrix V^T A V (upper triangle valid) */
  DenseMatrix get_T() { return T_.view(); }

  /** @brief Return the Cholesky factor of the most recent block */
  DenseMatrix get_beta() { return beta(); }

private:
  /** @brief Orthonormalizes @p v in place, leaving the Cholesky factor in beta.
   *
   *  v_new = v * R^{-1} where R^T R = v^T v, so v_old = v_new * R.
   */
  void orthonormalize(Panel v)
  {
    op->dot(v, v, gram_);
    gram_.view().cholesky_inverse(beta(), beta_inv_, status_ptr());
    v.mult(TransposeMode::NoTranspose, beta_inv_, v);
  }

  /** @brief Rayleigh-Ritz on the projected matrix; returns the converged count.
   *
   *  This moves T to the host and solves the small eigenproblem there.
   */
  unsigned int solve_small_dense()
  {
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    const std::size_t n_total = ncv;

    // Step i wrote rows 0..i of column i, so the device buffer is row-major
    // with only the upper triangle valid. Read column-major it is the
    // transpose, whose lower triangle is those same entries -- and the lower
    // triangle is the only part SelfAdjointEigenSolver references. So no
    // mirroring and no copy: the untouched half of the buffer lands in the half
    // that is never read.
    Eigen::SelfAdjointEigenSolver<Matrix> solver;
    {
      auto T_host = backend.host_block(T_, Access::Read);
      solver.compute(Eigen::Map<const Matrix>(T_host.data(), n_total, n_total));
    }
    if (solver.info() != Eigen::Success) throw std::runtime_error("Eigendecomposition failed");

    // Reorder the eigenvalues in descending order (in terms of absolute value)
    std::vector<unsigned int> indices(n_total);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](const auto& i, const auto& j) { return std::abs(solver.eigenvalues()[i]) > std::abs(solver.eigenvalues()[j]); });

    eigenvalues.resize(n_total);
    for (std::size_t i = 0; i < n_total; ++i) eigenvalues[i] = solver.eigenvalues()(indices[i]);

    // Move the Cholesky factor beta to the host (it is needed to estimate the residuals below)
    std::array<Scalar, blocksize * blocksize> beta_host{};
    {
      auto bstat_host = backend.host_block(bstat_, Access::Read);
      if (bstat_host[blocksize * blocksize] != Scalar{0}) throw std::runtime_error("Cholesky factorization failed in orthonormalize");
      std::copy_n(bstat_host.data(), blocksize * blocksize, beta_host.begin());
    }

    // Residual norm of Ritz pair j is ||beta * y_j||, over the last blocksize components of the eigenvector
    std::size_t n_converged = 0;
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    const std::size_t n_check = std::min<std::size_t>(nev, n_total);

    for (std::size_t col = 0; col < n_check; ++col) {
      Scalar norm_sq = 0;
      for (unsigned int i = 0; i < blocksize; ++i) {
        Scalar sum = 0;
        for (unsigned int j = 0; j < blocksize; ++j) sum += beta_host[i * blocksize + j] * solver.eigenvectors()(n_total - blocksize + j, indices[col]);
        norm_sq += sum * sum;
      }

      const Scalar denom = std::max(std::abs(eigenvalues[col]), eps);
      if (std::sqrt(norm_sq) / denom < tolerance) n_converged++;
    }

    if (n_converged < nev) prepare_restart(solver.eigenvectors(), indices);

    return n_converged;
  }

  /** @brief Writes the restarted T */
  template <class Eigenvectors>
  void prepare_restart(const Eigenvectors& evecs, const std::vector<unsigned int>& indices)
  {
    const std::size_t n_total = ncv;

    // Here we only put the Ritz values on the diagonal of T; the rest of the "arrowhead"
    // structure is put into T in extend(), see the commend in solve()
    {
      auto T_host = backend.host_block(T_, Access::Write);
      std::fill_n(T_host.data(), n_total * n_total, Scalar{0});
      for (std::size_t j = 0; j < nev; ++j) T_host[j * n_total + j] = eigenvalues[j];
    }

    {
      auto YY_host = backend.host_block(YY_, Access::Write);
      for (std::size_t r = 0; r < n_total; ++r)
        for (std::size_t c = 0; c < nev; ++c) YY_host[r * nev + c] = evecs(r, indices[c]);
    }
  }

  DenseMatrix beta() { return bstat_.view().block(0, 0, blocksize, blocksize); }
  DenseMatrix status_view() { return bstat_.view().block(blocksize, 0, 1, blocksize); }
  Scalar* status_ptr() { return bstat_.data() + blocksize * blocksize; }

  // Parameters
  unsigned int nev;
  unsigned int ncv;
  unsigned int max_restarts;
  double tolerance;

  std::vector<Scalar> eigenvalues;

  // Basis, and the restart target for the Ritz vectors
  BMV V;
  BMV W;

  typename B::OwnedDenseMatrix T_;        // ncv x ncv, the projected matrix V^T A V
  typename B::OwnedDenseMatrix YY_;       // ncv x nev, the kept Ritz coefficients
  typename B::OwnedDenseMatrix H2_;       // scratch space for the orthogonalization routine (so we don't have to allocate there)
  typename B::OwnedDenseMatrix gram_;     // blocksize x blocksize
  typename B::OwnedDenseMatrix beta_inv_; // blocksize x blocksize
  typename B::OwnedDenseMatrix bstat_;    // beta, with the Cholesky status in the trailing row

  Reorth reorth_{};

  std::shared_ptr<O> op;
  [[no_unique_address]] B backend;
};
} // namespace trl
