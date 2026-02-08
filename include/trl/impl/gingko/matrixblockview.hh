#pragma once

#include <ginkgo/ginkgo.hpp>

#include <exception>

namespace trl::ginkgo {
template <class T, unsigned int bs>
class MatrixBlockView {
public:
  using EntryType = T;

  MatrixBlockView(gko::matrix::Dense<T>* data)
      : data_(data)
  {
  }

  void copy_from(MatrixBlockView other) { *data_ = *(other.data_); }
  void copy_from_transpose(MatrixBlockView other) { other.data_->transpose(data_); }

  void set_zero() { data_->fill(0); }

  void set_diagonal(const gko::array<T>& diag)
  {
    if (std::dynamic_pointer_cast<const gko::OmpExecutor>(data_->get_executor()))
      for (unsigned int i = 0; i < bs; ++i) data_->at(i, i) = diag.get_const_data()[i];
    else throw std::runtime_error("set_diagonal is only implemented for the OMP executor");
  }

  void mult(MatrixBlockView X, MatrixBlockView Y) { data_->apply(X.data_, Y.data_); }

  unsigned int rows() const { return bs; }
  unsigned int cols() const { return bs; }

  gko::matrix::Dense<T>* data() { return data_; }

private:
  gko::matrix::Dense<T>* data_;
  template <class, unsigned int>
  friend class BlockView;
};
} // namespace trl::ginkgo
