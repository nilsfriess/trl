#pragma once

#include <cmath>
#include <cstddef>
#include <cstdlib>

#include "trl/helpers.hh"

namespace trl::openmp {
/** @brief Non-owning view of a row-major dense matrix in host memory.
 *
 *  This class does not parallelise anything using #pragma omp parallel
 *  or similar because we expect DenseMatrix to be small (maybe up to 64x64).
 */
template <class T>
class DenseMatrix {
public:
  DenseMatrix(T* data, unsigned int rows, unsigned int cols)
      : rows_(rows)
      , cols_(cols)
      , ld_(cols)
      , data_(data)
  {
  }

  DenseMatrix(T* data, unsigned int rows, unsigned int cols, unsigned int ld)
      : rows_(rows)
      , cols_(cols)
      , ld_(ld)
      , data_(data)
  {
  }

  /** @brief The nr x nc sub-matrix whose top-left entry is (r, c).
   *
   *  Shares the leading dimension, so the result stays a view into the same
   *  storage.
   */
  DenseMatrix block(unsigned int r, unsigned int c, unsigned int nr, unsigned int nc) const
  {
    TRL_CHECK(r + nr <= rows_ && c + nc <= cols_, "DenseMatrix::block out of range");
    return {data_ + std::size_t(r) * ld_ + c, nr, nc, ld_};
  }

  T* data() const { return data_; }

  /** @brief Computes *this += other */
  void add(const DenseMatrix& other)
  {
    TRL_CHECK(rows() == other.rows() && cols() == other.cols(), "DenseMatrix::add: shapes must match");

    for (unsigned int r = 0; r < rows_; ++r)
      for (unsigned int c = 0; c < cols_; ++c) data_[std::size_t(r) * ld_ + c] += other.data_[std::size_t(r) * other.ld_ + c];
  }

  /** @brief Zeroes the matrix, which may be a strided view (so memset is not enough). */
  void fill_zero()
  {
    for (unsigned int r = 0; r < rows_; ++r)
      for (unsigned int c = 0; c < cols_; ++c) data_[std::size_t(r) * ld_ + c] = T{0};
  }

  /** @brief *this = other^T (they must not alias). */
  void copy_from_transpose(DenseMatrix other)
  {
    TRL_CHECK(rows() == other.cols() && cols() == other.rows(), "DenseMatrix::copy_from_transpose: shapes must match");

    for (unsigned int r = 0; r < rows_; ++r)
      for (unsigned int c = 0; c < cols_; ++c) data_[std::size_t(r) * ld_ + c] = other.data_[std::size_t(c) * other.ld_ + r];
  }

  /** @brief Cholesky factor and its inverse.
   *
   *  Writes the upper triangular R with *this = R^T R into @p R, and R^{-1} into
   *  @p Rinv. On a non-positive pivot it sets @p status to a non-zero value and
   *  leaves the remaining entries unspecified.
   */
  void cholesky_inverse(DenseMatrix<T> R, DenseMatrix<T> Rinv, T* status)
  {
    TRL_CHECK(rows() == cols(), "DenseMatrix::cholesky_inverse: G must be square");
    TRL_CHECK(R.rows() == rows() && R.cols() == cols(), "cholesky_inverse: R must match G");
    TRL_CHECK(Rinv.rows() == rows() && Rinv.cols() == cols(), "cholesky_inverse: Rinv must match G");

    const T* g = data();
    T* r = R.data();
    T* ri = Rinv.data();
    const auto n = rows();
    const auto ld_g = ld();
    const auto ld_r = R.ld();
    const auto ld_i = Rinv.ld();

    bool failed = false;

    for (unsigned int j = 0; j < n; ++j) {
      for (unsigned int i = 0; i < j; ++i) {
        T sum = g[i * ld_g + j];
        for (unsigned int k = 0; k < i; ++k) sum -= r[k * ld_r + i] * r[k * ld_r + j];
        r[i * ld_r + j] = sum / r[i * ld_r + i];
      }

      T diag = g[j * ld_g + j];
      for (unsigned int k = 0; k < j; ++k) diag -= r[k * ld_r + j] * r[k * ld_r + j];

      if (!(diag > T{0})) {
        failed = true;
        diag = T{1}; // keep the remaining arithmetic finite; status flags the failure
      }
      r[j * ld_r + j] = std::sqrt(diag);

      for (unsigned int i = j + 1; i < n; ++i) r[i * ld_r + j] = T{0};
    }

    // Back substitution for the inverse of an upper triangular matrix.
    for (unsigned int j = 0; j < n; ++j) {
      for (unsigned int i = j + 1; i < n; ++i) ri[i * ld_i + j] = T{0};

      ri[j * ld_i + j] = T{1} / r[j * ld_r + j];

      for (unsigned int i = j; i-- > 0;) {
        T sum{0};
        for (unsigned int k = i + 1; k <= j; ++k) sum += r[i * ld_r + k] * ri[k * ld_i + j];
        ri[i * ld_i + j] = -sum / r[i * ld_r + i];
      }
    }

    if (failed) *status = T{1};
  }

  unsigned int rows() const { return rows_; }
  unsigned int cols() const { return cols_; }
  unsigned int ld() const { return ld_; }

  /** @brief Whether the entries are contiguous, i.e. safe to memcpy/memset. */
  bool contiguous() const { return ld_ == cols_; }

  std::size_t size() const { return std::size_t(rows_) * cols_; }

private:
  unsigned int rows_;
  unsigned int cols_;
  unsigned int ld_;
  T* data_;
};

/** @brief Owning, move-only host storage for a dense matrix.
 *
 *  Views are handed out by view(); the implicit conversion lets an owner be
 *  passed anywhere a DenseMatrix is expected.
 */
template <class T>
class OwnedDenseMatrix {
public:
  OwnedDenseMatrix(unsigned int rows, unsigned int cols)
      : rows_(rows)
      , cols_(cols)
  {
    const std::size_t bytes = std::size_t(rows) * cols * sizeof(T);
    const std::size_t aligned_bytes = (bytes + 63u) & ~std::size_t(63u);
    data_ = static_cast<T*>(std::aligned_alloc(64, aligned_bytes));
    for (std::size_t i = 0; i < std::size_t(rows) * cols; ++i) data_[i] = T{0};
  }

  ~OwnedDenseMatrix() { std::free(data_); }

  OwnedDenseMatrix(const OwnedDenseMatrix&) = delete;
  OwnedDenseMatrix& operator=(const OwnedDenseMatrix&) = delete;

  OwnedDenseMatrix(OwnedDenseMatrix&& other) noexcept
      : rows_(other.rows_)
      , cols_(other.cols_)
      , data_(other.data_)
  {
    other.data_ = nullptr;
  }

  OwnedDenseMatrix& operator=(OwnedDenseMatrix&& other) noexcept
  {
    if (this != &other) {
      std::free(data_);
      rows_ = other.rows_;
      cols_ = other.cols_;
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }

  DenseMatrix<T> view() const { return {data_, rows_, cols_}; }
  operator DenseMatrix<T>() const { return view(); }

  T* data() const { return data_; }
  unsigned int rows() const { return rows_; }
  unsigned int cols() const { return cols_; }

private:
  unsigned int rows_;
  unsigned int cols_;
  T* data_;
};
} // namespace trl::openmp
