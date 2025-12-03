#pragma once

#include "matrixblockview.hh"

#include <cstddef>
#include <memory>

#include <ginkgo/ginkgo.hpp>

namespace trl::ginkgo {
template <class T, unsigned int bs>
class BlockMatrix {
public:
  using BlockView = MatrixBlockView<T, bs>;

  BlockMatrix(std::shared_ptr<const gko::Executor> exec, std::size_t block_rows, std::size_t block_cols)
      : exec_(std::move(exec))
      , block_rows_(block_rows)
      , block_cols_(block_cols)
  {
    data = gko::array<T>(exec_, block_rows * block_cols * bs * bs);
    block_views.reserve(block_rows);
    for (std::size_t block_row = 0; block_row < block_rows; ++block_row) {
      std::vector<std::unique_ptr<gko::matrix::Dense<T>>> curr_views;
      curr_views.reserve(block_cols);
      for (std::size_t block_col = 0; block_col < block_rows; ++block_col) {
        auto block_idx = block_col * block_rows + block_row;
        auto view = gko::array<T>::view(exec_, bs * bs, data.get_data() + block_idx * bs * bs);
        curr_views.emplace_back(gko::matrix::Dense<T>::create(exec_, {bs, bs}, std::move(view), 0));
      }

      block_views.emplace_back(std::move(curr_views));
    }
  }

  std::size_t block_rows() const { return block_rows_; }
  std::size_t block_cols() const { return block_cols_; }

  BlockView block_view(std::size_t row, std::size_t col) const { return BlockView(block_views[row][col].get()); }

private:
  std::shared_ptr<const gko::Executor> exec_;
  std::size_t block_rows_;
  std::size_t block_cols_;

  gko::array<T> data;
  std::vector<std::vector<std::unique_ptr<gko::matrix::Dense<T>>>> block_views;
};
} // namespace trl::ginkgo
