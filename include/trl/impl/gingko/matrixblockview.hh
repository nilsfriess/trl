#pragma once

#include <ginkgo/ginkgo.hpp>

#include <exception>

namespace trl::ginkgo {
template <class T, unsigned int bs>
class MatrixBlockView {
public:
  using EntryType = T;

  MatrixBlockView(gko::matrix::Dense<T>* data)
      : data(data)
  {
  }

  void copy_from(MatrixBlockView other) { *data = *(other.data); }
  void copy_from_transpose(MatrixBlockView other) { other.data->transpose(data); }

  void set_zero() { data->fill(0); }

  void set_diagonal(const gko::array<T>& diag)
  {
    if (std::dynamic_pointer_cast<const gko::OmpExecutor>(data->get_executor()))
      for (unsigned int i = 0; i < bs; ++i) data->at(i, i) = diag.get_const_data()[i];
    else throw std::runtime_error("set_diagonal is only implemented for the OMP executor");
  }

  void mult(MatrixBlockView X, MatrixBlockView Y) { data->apply(X.data, Y.data); }

  gko::matrix::Dense<T>* data;

private:
  template <class, unsigned int>
  friend class BlockView;
};
} // namespace trl::ginkgo
