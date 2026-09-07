#pragma once

#include <sycl/sycl.hpp>

namespace trl::Sycl {
template <class T>
class DenseMatrix {
public:
  DenseMatrix(sycl::queue q, unsigned int rows, unsigned int cols)
      : rows_(rows)
      , cols_(cols)
      , data_(sycl::malloc_device<T>(rows * cols, q))
  {
  }

  // Non-owning
  DenseMatrix(T* data, unsigned int rows, unsigned int cols)
      : rows_(rows)
      , cols_(cols)
      , data_(data)
  {
  }

  T* data() { return data_; }
  const T* data() const { return data_; }

  unsigned int rows() const { return rows_; }
  unsigned int cols() const { return cols_; }

private:
  // sycl::queue q;
  unsigned int rows_;
  unsigned int cols_;
  T* data_;
};
} // namespace trl::Sycl
