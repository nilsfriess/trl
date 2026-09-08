#pragma once

#include <cstddef>
#include <utility>

#include <sycl/sycl.hpp>

#include "trl/helpers.hh"

namespace trl::Sycl {
/** @brief Non-owning view of a row-major dense matrix in device memory.
 *
 *  Carries a leading dimension, so a view can name a sub-block of a larger
 *  matrix: the Lanczos coefficient vector for step i is the
 *  (i+1)*bs x bs sub-block of the projected matrix T at column i*bs, and every
 *  kernel that touches it needs a row stride independent of its column count.
 */
template <class T>
class DenseMatrix {
public:
  DenseMatrix(sycl::queue* q, T* data, unsigned int rows, unsigned int cols)
      : q(q)
      , rows_(rows)
      , cols_(cols)
      , ld_(cols)
      , data_(data)
  {
  }

  DenseMatrix(sycl::queue* q, T* data, unsigned int rows, unsigned int cols, unsigned int ld)
      : q(q)
      , rows_(rows)
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
    return {q, data_ + std::size_t(r) * ld_ + c, nr, nc, ld_};
  }

  T* data() const { return data_; }

  /** @brief Computes *this += other */
  void add(const DenseMatrix& other)
  {
    TRL_CHECK(rows() == other.rows() && cols() == other.cols(), "DenseMatrix::add: shapes must match");

    T* d = data();
    const T* s = other.data();
    const auto n_cols = cols();
    const auto ld_d = ld();
    const auto ld_s = other.ld();
    q->parallel_for(sycl::range<1>(size()), [d, s, n_cols, ld_d, ld_s](sycl::id<1> id) {
      const auto k = id[0];
      const auto r = k / n_cols;
      const auto c = k % n_cols;
      d[r * ld_d + c] += s[r * ld_s + c];
    });
  }

  /** @brief Zeroes @p M, which may be a strided view (so memset is not enough). */
  sycl::event fill_zero()
  {
    if (contiguous()) return q->memset(data(), 0, size() * sizeof(T));

    T* d = data();
    const auto n_cols = cols();
    const auto n_ld = ld();
    return q->parallel_for(sycl::range<1>(size()), [=](sycl::id<1> id) {
      const auto k = id[0];
      d[(k / n_cols) * n_ld + (k % n_cols)] = T{0};
    });
  }

  /** @brief *this = other^T (they must not alias). */
  void copy_from_transpose(DenseMatrix other)
  {
    TRL_CHECK(rows() == other.cols() && cols() == other.rows(), "DenseMatrix::copy_from_transpose: shapes must match");

    const T* s = other.data();
    T* d = data();
    const auto n_cols = cols();
    const auto ld_d = ld();
    const auto ld_s = other.ld();
    q->parallel_for(sycl::range<1>(size()), [d, s, n_cols, ld_d, ld_s](sycl::id<1> id) {
      const auto k = id[0];
      const auto r = k / n_cols;
      const auto c = k % n_cols;
      d[r * ld_d + c] = s[c * ld_s + r];
    });
  }

  /** @brief Cholesky factor and its inverse, on the device.
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

    q->single_task([=] {
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
        r[j * ld_r + j] = sycl::sqrt(diag);

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
    });
  }

  unsigned int rows() const { return rows_; }
  unsigned int cols() const { return cols_; }
  unsigned int ld() const { return ld_; }

  /** @brief Whether the entries are contiguous, i.e. safe to memcpy/memset. */
  bool contiguous() const { return ld_ == cols_; }

  std::size_t size() const { return std::size_t(rows_) * cols_; }

private:
  sycl::queue* q;
  unsigned int rows_;
  unsigned int cols_;
  unsigned int ld_;
  T* data_;
};

/** @brief Owning, move-only device storage for a dense matrix.
 *
 *  Views are handed out by view(); the implicit conversion lets an owner be
 *  passed anywhere a DenseMatrix is expected.
 */
template <class T>
class OwnedDenseMatrix {
public:
  OwnedDenseMatrix(sycl::queue queue, unsigned int rows, unsigned int cols)
      : queue_(queue)
      , rows_(rows)
      , cols_(cols)
      , data_(sycl::malloc_device<T>(std::size_t(rows) * cols, queue))
  {
    queue_.memset(data_, 0, std::size_t(rows) * cols * sizeof(T)).wait();
  }

  ~OwnedDenseMatrix()
  {
    if (data_) sycl::free(data_, queue_);
  }

  OwnedDenseMatrix(const OwnedDenseMatrix&) = delete;
  OwnedDenseMatrix& operator=(const OwnedDenseMatrix&) = delete;

  OwnedDenseMatrix(OwnedDenseMatrix&& other) noexcept
      : queue_(std::move(other.queue_))
      , rows_(other.rows_)
      , cols_(other.cols_)
      , data_(other.data_)
  {
    other.data_ = nullptr;
  }

  OwnedDenseMatrix& operator=(OwnedDenseMatrix&& other) noexcept
  {
    if (this != &other) {
      if (data_) sycl::free(data_, queue_);
      queue_ = std::move(other.queue_);
      rows_ = other.rows_;
      cols_ = other.cols_;
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }

  DenseMatrix<T> view() const { return {&queue_, data_, rows_, cols_}; }
  operator DenseMatrix<T>() const { return view(); }

  T* data() const { return data_; }
  unsigned int rows() const { return rows_; }
  unsigned int cols() const { return cols_; }

private:
  mutable sycl::queue queue_;
  unsigned int rows_;
  unsigned int cols_;
  T* data_;
};
} // namespace trl::Sycl
