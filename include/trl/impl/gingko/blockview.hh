#pragma once

#include "matrixblockview.hh"

#include <ginkgo/ginkgo.hpp>

namespace trl::ginkgo {
template <class T, unsigned int bs>
class BlockView {
public:
  using EntryType = T;
  using MatrixBlockView = MatrixBlockView<T, bs>;

  BlockView(gko::matrix::Dense<T>* data)
      : one(gko::matrix::Dense<T>::create(data->get_executor(), gko::dim<2>{1, 1}))
      , minus_one(gko::matrix::Dense<T>::create(data->get_executor(), gko::dim<2>{1, 1}))
      , data(data)
  {
    one->fill(1);
    minus_one->fill(-1);
  }

  void copy_from(BlockView other) { *data = *(other.data); }

  BlockView& operator-=(BlockView other)
  {
    data->add_scaled(minus_one, other.data);
    return *this;
  }

  void mult_add(MatrixBlockView W, BlockView other) { other.data->apply(one, W.data, one, other.data); }

  void mult(MatrixBlockView W, BlockView other) { data->apply(W.data, other.data); }

  void mult_transpose(MatrixBlockView W, BlockView other)
  {
    auto Wt = W.data->transpose();
    data->apply(Wt, other.data);
  }

  void dot(BlockView other, MatrixBlockView B) { data->compute_dot(other.data, B.data); }

  void subtract_product(BlockView other, MatrixBlockView B) { other.data->apply(minus_one, other.data, one, data); }

  void set_zero() { data->fill(0); }

  std::size_t rows() const { return data->get_size()[0]; }
  std::size_t cols() const { return data->get_size()[1]; }

private:
  std::shared_ptr<gko::matrix::Dense<T>> one;
  std::shared_ptr<gko::matrix::Dense<T>> minus_one;
  gko::matrix::Dense<T>* data;
};
} // namespace trl::ginkgo
