#pragma once

#include "trl/concepts.hh"
#include "trl/helpers.hh"
#include "trl/impl/openmp/dense_matrix.hh"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace trl::openmp {
/** @brief A view of `count` consecutive blocks of a BlockMultivector.
 */
template <class T, unsigned int bs>
class PanelView {
public:
  static constexpr unsigned int blocksize = bs;

  PanelView(T* start, std::size_t rows, unsigned int count)
      : start_(start)
      , rows_(rows)
      , count_(count)
  {
  }

  /** @brief The single-block panel for block @p i of this panel. */
  PanelView block(unsigned int i) const
  {
    TRL_CHECK(i < count_, "Block index out of range");
    return {start_ + std::size_t(i) * rows_ * bs, rows_, 1};
  }

  void copy_from(PanelView other)
  {
    TRL_CHECK(rows() == other.rows(), "The number of rows of panels must match to copy them");
    TRL_CHECK(cols() == other.cols(), "The number of columns of panels must match to copy them");

    T* d = data();
    const T* s = other.data();
    const std::size_t total = rows() * cols();

#pragma omp parallel for
    for (std::size_t i = 0; i < total; ++i) d[i] = s[i];
  }

  /** @brief Computes out = this * M (or this * M^T for TransposeMode::Transpose).
   *
   *  @p M is (cols() x out.cols()) row-major for NoTranspose and
   *  (out.cols() x cols()) row-major for TransposeMode::Transpose, i.e. the
   *  transpose factor itself is stored rather than computed.
   */
  void mult(TransposeMode t, DenseMatrix<T> M, PanelView out)
  {
    TRL_CHECK(rows() == out.rows(), "Input panel and output panel must have the same number of rows");
    if (t == TransposeMode::NoTranspose) {
      TRL_CHECK(cols() == M.rows(), "Number of columns of input panel must match number of rows of matrix");
      TRL_CHECK(M.cols() == out.cols(), "Number of columns of output panel must match number of cols of matrix");
    }
    else {
      TRL_CHECK(out.cols() == M.rows(), "For TransposeMode::Transpose the matrix must be stored as (out.cols() x cols())");
      TRL_CHECK(M.cols() == cols(), "For TransposeMode::Transpose the number of columns of the matrix must match the number of columns of the input panel");
    }

    // When in and out alias, the loop below only works for single-block panels
    if (out.data() == data()) TRL_CHECK(count_ == 1 && out.count_ == 1, "In-place mult is only supported for single-block panels");

    const T* a = data();
    const T* b = M.data();
    T* c = out.data();
    const auto in_blocks = count_;
    const auto out_blocks = out.count_;
    const auto n = rows();
    const auto ldb = M.ld();

    // Both storage conventions are the same walk over M with the two strides
    // swapped, so the transpose flag never reaches the inner loop.
    const bool tp = (t == TransposeMode::Transpose);
    const std::size_t si = tp ? ldb : 1;
    const std::size_t sj = tp ? 1 : ldb;

#pragma omp parallel for
    for (std::size_t tid = 0; tid < n; ++tid) {
      // Output blocks are processed in tiles rather than one at a time, so
      // the input panel is read ceil(out_blocks / tile) times instead of
      // out_blocks times. That matters for the restart Ritz product, where
      // the input panel is the whole basis.
      for (unsigned int Bo0 = 0; Bo0 < out_blocks; Bo0 += tile) {
        const unsigned int n_out = std::min(tile, out_blocks - Bo0);

        alignas(64) T c_private[tile * bs];
        for (unsigned int i = 0; i < tile * bs; ++i) c_private[i] = T{0};

        for (unsigned int Bi = 0; Bi < in_blocks; ++Bi) {
          const T* a_base = a + std::size_t(Bi) * n * bs;

          alignas(64) T a_private[bs];
          for (unsigned int j = 0; j < bs; ++j) a_private[j] = a_base[tid * bs + j];

          for (unsigned int t_ = 0; t_ < n_out; ++t_) {
            const unsigned int Bo = Bo0 + t_;
            const T* b_blk = b + (tp ? std::size_t(Bo) * bs * ldb + std::size_t(Bi) * bs : std::size_t(Bi) * bs * ldb + std::size_t(Bo) * bs);

            for (unsigned int i = 0; i < bs; ++i) {
              T sum{0};
#pragma omp simd reduction(+ : sum)
              for (unsigned int j = 0; j < bs; ++j) sum += a_private[j] * b_blk[i * si + j * sj];
              c_private[t_ * bs + i] += sum;
            }
          }
        }

        for (unsigned int t_ = 0; t_ < n_out; ++t_) {
          T* c_base = c + std::size_t(Bo0 + t_) * n * bs;
          for (unsigned int i = 0; i < bs; ++i) c_base[tid * bs + i] = c_private[t_ * bs + i];
        }
      }
    }
  }

  void subtract(PanelView other)
  {
    TRL_CHECK(rows() == other.rows() && cols() == other.cols(), "PanelView::subtract: shapes must match");

    T* a = data();
    const T* b = other.data();
    const std::size_t total = rows() * cols();

#pragma omp parallel for
    for (std::size_t i = 0; i < total; ++i) a[i] -= b[i];
  }

  /** @brief Computes out = this^T * Y, with Y a single block.
   *
   *  @p out is (cols() x Y.cols()) and may be strided, so that the coefficients
   *  can be written straight into a column of the projected matrix.
   *
   *  Each thread owns a contiguous chunk of rows and accumulates the whole
   *  cols() x bs coefficient matrix privately; the chunks are combined once at
   *  the end. Splitting the rows rather than the blocks keeps every read
   *  sequential and the reduction out of the inner loop.
   */
  void dot(PanelView Y, DenseMatrix<T> out)
  {
    TRL_CHECK(rows() == Y.rows(), "Both panels must have the same number of rows");
    TRL_CHECK(Y.count_ == 1, "The right operand of dot must be a single block");
    TRL_CHECK(out.rows() == cols(), "The output must have one row per column of this panel");
    TRL_CHECK(out.cols() == Y.cols(), "The output must have one column per column of the right operand");

    out.fill_zero();

    const T* a = data();
    const T* b = Y.data();
    T* c = out.data();
    const auto ldc = out.ld();
    const auto n_blocks = count_;
    const auto n = rows();

#pragma omp parallel
    {
      std::vector<T> sums(std::size_t(n_blocks) * bs * bs, T{0});

#pragma omp for
      for (std::size_t i = 0; i < n; ++i) {
        alignas(64) T b_private[bs];
        for (unsigned int J = 0; J < bs; ++J) b_private[J] = b[i * bs + J];

        for (unsigned int B = 0; B < n_blocks; ++B) {
          const T* a_block = a + std::size_t(B) * n * bs;
          T* s = sums.data() + std::size_t(B) * bs * bs;

          for (unsigned int I = 0; I < bs; ++I) {
            const T a_val = a_block[i * bs + I];
#pragma omp simd
            for (unsigned int J = 0; J < bs; ++J) s[I * bs + J] += a_val * b_private[J];
          }
        }
      }

#pragma omp critical
      {
        for (unsigned int B = 0; B < n_blocks; ++B)
          for (unsigned int I = 0; I < bs; ++I)
            for (unsigned int J = 0; J < bs; ++J) c[std::size_t(B * bs + I) * ldc + J] += sums[(std::size_t(B) * bs + I) * bs + J];
      }
    }
  }

  /** @brief Computes this -= Y * M (or this -= Y * M^T for TransposeMode::Transpose),
   *  the fused form of mult + subtract.
   *
   *  @p M is (Y.cols() x cols()) row-major for NoTranspose and
   *  (cols() x Y.cols()) row-major for TransposeMode::Transpose, i.e. the
   *  transpose factor itself is stored rather than computed.
   */
  void subtract_product(TransposeMode t, PanelView Y, DenseMatrix<T> M)
  {
    TRL_CHECK(rows() == Y.rows(), "This panel and the input panel must have the same number of rows");
    if (t == TransposeMode::NoTranspose) {
      TRL_CHECK(Y.cols() == M.rows(), "Number of columns of the input panel must match number of rows of matrix");
      TRL_CHECK(M.cols() == cols(), "Number of columns of matrix must match number of columns of this panel");
    }
    else {
      TRL_CHECK(cols() == M.rows(), "For TransposeMode::Transpose the matrix must be stored as (cols() x Y.cols())");
      TRL_CHECK(M.cols() == Y.cols(), "For TransposeMode::Transpose the number of columns of the matrix must match the number of columns of the input panel");
    }

    const T* a = Y.data();
    T* c = data();
    const T* b = M.data();
    const auto in_blocks = Y.count_;
    const auto out_blocks = count_;
    const auto n = rows();
    const auto ldb = M.ld();

    const bool tp = (t == TransposeMode::Transpose);
    const std::size_t si = tp ? ldb : 1;
    const std::size_t sj = tp ? 1 : ldb;

#pragma omp parallel for
    for (std::size_t tid = 0; tid < n; ++tid) {
      for (unsigned int Bo = 0; Bo < out_blocks; ++Bo) {
        alignas(64) T c_private[bs];
        for (unsigned int i = 0; i < bs; ++i) c_private[i] = T{0};

        for (unsigned int Bi = 0; Bi < in_blocks; ++Bi) {
          const T* a_base = a + std::size_t(Bi) * n * bs;

          alignas(64) T a_private[bs];
          for (unsigned int j = 0; j < bs; ++j) a_private[j] = a_base[tid * bs + j];

          const T* b_blk = b + (tp ? std::size_t(Bo) * bs * ldb + std::size_t(Bi) * bs : std::size_t(Bi) * bs * ldb + std::size_t(Bo) * bs);

          for (unsigned int i = 0; i < bs; ++i) {
            T sum{0};
#pragma omp simd reduction(+ : sum)
            for (unsigned int j = 0; j < bs; ++j) sum += a_private[j] * b_blk[i * si + j * sj];
            c_private[i] += sum;
          }
        }

        T* c_base = c + std::size_t(Bo) * n * bs;
        for (unsigned int i = 0; i < bs; ++i) c_base[tid * bs + i] -= c_private[i];
      }
    }
  }

  T* data() const { return start_; }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return std::size_t(count_) * bs; }
  unsigned int blocks() const { return count_; }

private:
  /** @brief Output blocks held in registers at once by mult, kept at
   *  tile * bs accumulators regardless of the block size. */
  static constexpr unsigned int tile = bs >= 8 ? 1u : (bs >= 4 ? 2u : 8u);

  T* start_;
  std::size_t rows_;
  unsigned int count_;
};
} // namespace trl::openmp
