#pragma once

#include "trl/concepts.hh"
#include "trl/helpers.hh"
#include "trl/impl/sycl/dense_matrix.hh"
#include "trl/impl/sycl/launch_config.hh"

#include <cstddef>
#include <sycl/sycl.hpp>
#include <vector>

namespace trl::Sycl {
/** @brief A view of `count` consecutive blocks of a BlockMultivector.
 *
 *  The unit the Lanczos iteration actually works in: a panel is a block of
 *  blocks, and a single block is just the count == 1 case. Every operation
 *  parallelises over the rows, never over the block count, so the launch
 *  geometry is independent of how wide the panel is.
 */
template <class T, unsigned int bs>
class PanelView {
public:
  static constexpr unsigned int blocksize = bs;

  /** @param launch Launch geometry for dot, derived from the device once by the
   *  owning BlockMultivector and handed to every view. */
  PanelView(sycl::queue* q_, T* start, std::size_t rows, unsigned int count, DotLaunch launch)
      : q(q_)
      , start_(start)
      , rows_(rows)
      , count_(count)
      , launch_(launch)
  {
  }

  /** @brief The single-block panel for block @p i of this panel. */
  PanelView block(unsigned int i) const
  {
    TRL_CHECK(i < count_, "Block index out of range");
    return {q, start_ + std::size_t(i) * rows_ * bs, rows_, 1, launch_};
  }

  void copy_from(PanelView other)
  {
    TRL_CHECK(rows() == other.rows(), "The number of rows of panels must match to copy them");
    TRL_CHECK(cols() == other.cols(), "The number of rows of panels must match to copy them");
    q->memcpy(data(), other.data(), rows() * cols() * sizeof(T));
  }

  /** @brief Computes out = this * M (or this * M^T for TransposeMode::Transpose).
   *
   *  @p M is (cols() x out.cols()) row-major for NoTranspose and
   *  (out.cols() x cols()) row-major for TransposeMode::Transpose, i.e. the
   *  transpose factor itself is stored rather than computed. If @p events is
   *  given, the events of all submitted kernels are appended to it, as in dot.
   */
  void mult(TransposeMode t, DenseMatrix<T> M, PanelView out, std::vector<sycl::event>* events = nullptr)
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

    // When in and out alias, the kernel below only works for single-block panels
    if (out.data() == data()) TRL_CHECK(count_ == 1 && out.count_ == 1, "In-place mult is only supported for single-block panels");

    const T* a = data();
    const T* b = M.data();
    T* c = out.data();
    const auto in_blocks = count_;
    const auto out_blocks = out.count_;
    const auto n = rows();
    const auto ldb = M.ld();

    sycl::specialized<bool> tp = (t == TransposeMode::Transpose);

    sycl::event mult_event = q->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        auto tid = id[0];

        // Output blocks are processed in tiles rather than one at a time, so
        // the input panel is read ceil(out_blocks / tile) times instead of
        // out_blocks times. That matters for the restart Ritz product, where
        // the input panel is the whole basis.
        for (unsigned int Bo0 = 0; Bo0 < out_blocks; Bo0 += tile) {
          const unsigned int n_out = sycl::min(tile, out_blocks - Bo0);

          T c_private[tile * bs];
          for (unsigned int i = 0; i < tile * bs; ++i) c_private[i] = T{0};

          for (unsigned int Bi = 0; Bi < in_blocks; ++Bi) {
            const T* a_base = a + Bi * n * bs;

            T a_private[bs];
            for (unsigned int j = 0; j < bs; ++j) a_private[j] = a_base[tid * bs + j];

            for (unsigned int t_ = 0; t_ < n_out; ++t_)
              for (unsigned int i = 0; i < bs; ++i)
                for (unsigned int j = 0; j < bs; ++j)
                  if (tp) c_private[t_ * bs + i] += a_private[j] * b[((Bo0 + t_) * bs + i) * ldb + (Bi * bs + j)];
                  else c_private[t_ * bs + i] += a_private[j] * b[(Bi * bs + j) * ldb + ((Bo0 + t_) * bs + i)];
          }

          for (unsigned int t_ = 0; t_ < n_out; ++t_) {
            T* c_base = c + (Bo0 + t_) * n * bs;
            for (unsigned int i = 0; i < bs; ++i) c_base[tid * bs + i] = c_private[t_ * bs + i];
          }
        }
      });
    });
    if (events) events->push_back(mult_event);
  }

  void subtract(PanelView other)
  {
    const auto n = rows();

    q->submit([&](auto& cgh) {
      auto* a = data();
      const auto* b = other.data();
      sycl::specialized<unsigned int> n_blocks(count_);

      cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        auto tid = id[0];
        for (unsigned int B = 0; B < n_blocks; ++B) {
          auto* a_start = a + B * n * bs;
          const auto* b_start = b + B * n * bs;
          for (unsigned int i = 0; i < bs; ++i) a_start[tid * bs + i] -= b_start[tid * bs + i];
        }
      });
    });
  }

  /** @brief Computes out = this^T * Y, with Y a single block.
   *
   *  @p out is (cols() x Y.cols()) and may be strided, so that the coefficients
   *  can be written straight into a column of the projected matrix.
   *
   *  The parallelism comes from the rows: the grid is
   *  (row tiles) x (blocks of this panel), so it stays thousands of work-groups
   *  wide no matter how many blocks the panel holds, and each work item keeps
   *  only bs*bs accumulators. Parallelising over the block count instead would
   *  fill one or two wavefronts, serialise the reduction per lane, and make
   *  adjacent lanes touch addresses n*sizeof(T) apart.
   */
  void dot(PanelView Y, DenseMatrix<T> out, std::vector<sycl::event>* events = nullptr)
  {
    TRL_CHECK(rows() == Y.rows(), "Both panels must have the same number of rows");
    TRL_CHECK(Y.count_ == 1, "The right operand of dot must be a single block");
    TRL_CHECK(out.rows() == cols(), "The output must have one row per column of this panel");
    TRL_CHECK(out.cols() == Y.cols(), "The output must have one column per column of the right operand");

    sycl::event fill_event = out.fill_zero();
    if (events) events->push_back(fill_event);

    const auto local_size = launch_.local_size;
    const auto global_size = launch_.global_size();

    const T* a = data();
    const T* b = Y.data();
    T* c = out.data();
    const auto ldc = out.ld();
    const auto n = rows();

    sycl::specialized<bool> is_interleaved(launch_.interleaved);

    sycl::event reduce_event = q->submit([&](sycl::handler& cgh) {
      // Captured explicitly and widest-first: with an implicit [=] clang orders the
      // closure by first use in the body, and any 4- or 1-byte capture sitting before
      // a pointer leaves alignment padding that AdaptiveCpp emits as one kernel
      // parameter per padding byte.
      cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(global_size, count_), sycl::range<2>(local_size, 1)),
                       [a, b, c, n, ldc, is_interleaved](sycl::nd_item<2> it) {
        const std::size_t gid = it.get_global_id(0);
        const std::size_t gsize = it.get_global_range(0);
        const std::size_t B = it.get_global_id(1); // which block of the panel this group reduces

        const T* a_block = a + B * n * bs;

        // The two loops differ only in how rows are distributed; since is_interleaved is
        // of type sycl::specialized, AdaptiveCpp's JIT compiler will optimise the branch
        // away at JIT compile time.
        T sum[bs * bs] = {0};
        if (is_interleaved) {
          // GPU: consecutive work-items take consecutive rows (so the loads will
          // be coalesced)
          for (std::size_t i = gid; i < n; i += gsize)
            for (unsigned int I = 0; I < bs; ++I)
              for (unsigned int J = 0; J < bs; ++J) sum[I * bs + J] += a_block[i * bs + I] * b[i * bs + J];
        }
        else {
          // CPU: individual work items take consecutive rows
          const std::size_t chunk = (n + gsize - 1) / gsize;
          const std::size_t begin = sycl::min(gid * chunk, n);
          const std::size_t end = sycl::min(begin + chunk, n);
          for (std::size_t i = begin; i < end; ++i)
            for (unsigned int I = 0; I < bs; ++I)
              for (unsigned int J = 0; J < bs; ++J) sum[I * bs + J] += a_block[i * bs + I] * b[i * bs + J];
        }

        T reduced_sums[bs * bs];
        // Note: Do not nest this loop. A compiler bug in AdaptiveCpp miscompiles this (see https://github.com/AdaptiveCpp/AdaptiveCpp/issues/2224)
        for (unsigned int K = 0; K < bs * bs; ++K) {
          // Reduce local sums within our group
          reduced_sums[K] = sycl::reduce_over_group(it.get_group(), sum[K], sycl::plus<T>());
        }

        if (it.get_group().leader()) {
          for (unsigned int I = 0; I < bs; ++I)
            for (unsigned int J = 0; J < bs; ++J) {
              sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> c_ref(c[(B * bs + I) * ldc + J]);
              c_ref += reduced_sums[I * bs + J];
            }
        }
      });
    });
    if (events) events->push_back(reduce_event);
  }

  /** @brief Computes this -= Y * M (or this -= Y * M^T for TransposeMode::Transpose),
   *  the fused form of mult + subtract.
   *
   *  @p M is (Y.cols() x cols()) row-major for NoTranspose and
   *  (cols() x Y.cols()) row-major for TransposeMode::Transpose, i.e. the
   *  transpose factor itself is stored rather than computed -- the same
   *  conventions as mult. Avoids both the temporary output panel and the extra
   *  read pass that calling mult into scratch storage and subtracting would
   *  cost: each row of this is read and written exactly once. If @p events is
   *  given, the event of the submitted kernel is appended to it, as in mult.
   */
  void subtract_product(TransposeMode t, PanelView Y, DenseMatrix<T> M, std::vector<sycl::event>* events = nullptr)
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

    sycl::specialized<bool> tp = (t == TransposeMode::Transpose);

    sycl::event subtract_event = q->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        auto tid = id[0];

        for (unsigned int Bo = 0; Bo < out_blocks; ++Bo) {
          T c_private[bs];
          for (unsigned int i = 0; i < bs; ++i) c_private[i] = T{0};

          for (unsigned int Bi = 0; Bi < in_blocks; ++Bi) {
            const T* a_base = a + Bi * n * bs;

            T a_private[bs];
            for (unsigned int j = 0; j < bs; ++j) a_private[j] = a_base[tid * bs + j];

            for (unsigned int i = 0; i < bs; ++i)
              for (unsigned int j = 0; j < bs; ++j)
                if (tp) c_private[i] += a_private[j] * b[(Bo * bs + i) * ldb + (Bi * bs + j)];
                else c_private[i] += a_private[j] * b[(Bi * bs + j) * ldb + (Bo * bs + i)];
          }

          T* c_base = c + Bo * n * bs;
          for (unsigned int i = 0; i < bs; ++i) c_base[tid * bs + i] -= c_private[i];
        }
      });
    });
    if (events) events->push_back(subtract_event);
  }

  T* data() const { return start_; }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return std::size_t(count_) * bs; }
  unsigned int blocks() const { return count_; }

private:
  /** @brief Output blocks held in registers at once by mult, kept at
   *  tile * bs accumulators regardless of the block size. */
  static constexpr unsigned int tile = bs >= 8 ? 1u : (bs >= 4 ? 2u : 8u);

  sycl::queue* q;

  T* start_;
  std::size_t rows_;
  unsigned int count_;

  DotLaunch launch_;
};

} // namespace trl::Sycl
