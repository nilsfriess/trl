#pragma once

#include "blockmatrix.hh"
#include "blockview.hh"

#include <ginkgo/ginkgo.hpp>

#include <cstddef>
#include <exception>
#include <memory>

namespace trl::ginkgo {
template <class T, unsigned int bs>
class BlockMultivector {
public:
  using Scalar = double;
  static_assert(std::is_same_v<Scalar, T>);
  static constexpr unsigned int blocksize = bs;

  using BlockView = BlockView<Scalar, bs>;
  using BlockMatrix = BlockMatrix<Scalar, bs>;

  BlockMultivector(std::shared_ptr<const gko::Executor> exec, std::size_t rows, std::size_t cols)
      : exec_(std::move(exec))
      , rows_(rows)
      , blocks_(cols / blocksize)
  {
    if (cols % blocksize != 0) throw std::invalid_argument("Number of columns must be divisible by blocksize");

    data = gko::array<Scalar>(exec_, rows * cols);
    block_views.reserve(blocks_);
    for (std::size_t block = 0; block < blocks_; ++block) {
      auto view = gko::array<Scalar>::view(exec_, rows * blocksize, data.get_data() + block * rows * blocksize);
      block_views.emplace_back(gko::matrix::Dense<Scalar>::create(exec_, {rows, blocksize}, std::move(view), 0));
    }
  }

  BlockView block_view(std::size_t block) { return BlockView(block_views[block].get()); }

private:
  std::shared_ptr<const gko::Executor> exec_;
  std::size_t rows_;
  std::size_t blocks_;

  gko::array<Scalar> data;
  std::vector<std::unique_ptr<gko::matrix::Dense<Scalar>>> block_views;
};
}; // namespace trl::ginkgo
