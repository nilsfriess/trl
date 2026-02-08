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
      : one_(gko::matrix::Dense<T>::create(data->get_executor(), gko::dim<2>{1, 1}))
      , minus_one_(gko::matrix::Dense<T>::create(data->get_executor(), gko::dim<2>{1, 1}))
      , data_(data)
  {
    one_->fill(1);
    minus_one_->fill(-1);
  }

  void copy_from(BlockView other) { *data_ = *(other.data_); }

  BlockView& operator-=(BlockView other)
  {
    data_->add_scaled(minus_one_, other.data_);
    return *this;
  }

  void mult_add(MatrixBlockView W, BlockView other) { data_->apply(one_, W.data_, one_, other.data_); }

  void mult(MatrixBlockView W, BlockView other) { data_->apply(W.data_, other.data_); }

  void mult_transpose(MatrixBlockView W, BlockView other)
  {
    auto Wt = W.data_->transpose();
    data_->apply(Wt, other.data_);
  }

  void dot(BlockView other, MatrixBlockView B)
  {
    auto data_t = data_->conj_transpose();
    data_t->apply(other.data_, B.data_);
  }

  void subtract_product(BlockView other, MatrixBlockView B) { other.data_->apply(minus_one_, B.data_, one_, data_); }

  void set_zero() { data_->fill(0); }

  std::size_t rows() const { return data_->get_size()[0]; }
  std::size_t cols() const { return data_->get_size()[1]; }

  gko::matrix::Dense<T>* data() { return data_; };

private:
  std::shared_ptr<gko::matrix::Dense<T>> one_;
  std::shared_ptr<gko::matrix::Dense<T>> minus_one_;
  gko::matrix::Dense<T>* data_;
};
} // namespace trl::ginkgo
