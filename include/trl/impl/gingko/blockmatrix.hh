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
      , data(exec_, block_rows * block_cols * bs * bs)
  {
    data.fill(0);
    block_views.reserve(block_rows);
    for (std::size_t block_row = 0; block_row < block_rows; ++block_row) {
      std::vector<std::unique_ptr<gko::matrix::Dense<T>>> curr_views;
      curr_views.reserve(block_cols);
      for (std::size_t block_col = 0; block_col < block_cols; ++block_col) {
        auto block_idx = block_col * block_rows + block_row;
        auto view = gko::array<T>::view(exec_, bs * bs, data.get_data() + block_idx * bs * bs);
        curr_views.emplace_back(gko::matrix::Dense<T>::create(exec_, {bs, bs}, std::move(view), bs));
      }

      block_views.emplace_back(std::move(curr_views));
    }
  }

  std::size_t block_rows() const { return block_rows_; }
  std::size_t block_cols() const { return block_cols_; }

  BlockView block_view(std::size_t row, std::size_t col) const { return BlockView(block_views[row][col].get()); }

  void print(bool with_separator = true, bool scientific = true, int precision = 8, T tolerance = 1e-10) const
  {
    if (with_separator) {
      std::cout << "--------------------------------------------------\n";
      std::cout << "Block matrix of order " << block_rows_ << "x" << block_cols_ << "\n";
    }

    // Set output format for high precision
    std::ios_base::fmtflags original_flags = std::cout.flags();
    std::streamsize original_precision = std::cout.precision();

    if (scientific) std::cout << std::scientific << std::setprecision(precision);
    else std::cout << std::fixed << std::setprecision(precision);

    // Calculate field width based on precision and format
    int field_width = scientific ? precision + 7 : precision + 4; // Adjust for format overhead

    for (std::size_t i = 0; i < block_rows_; ++i) {
      // Print each row of blocks
      for (std::size_t bi = 0; bi < bs; ++bi) {
        for (std::size_t j = 0; j < block_cols_; ++j) {
          auto block = block_view(i, j);

          // Print vertical separator before block (except for first block)
          if (j > 0 && with_separator) std::cout << "|";

          for (std::size_t bj = 0; bj < bs; ++bj) {
            auto entry = block.data()->at(bi, bj);
            if (std::abs(entry) < tolerance) std::cout << std::setw(field_width) << "*";
            else std::cout << std::setw(field_width) << entry;
            if (bj < bs - 1) std::cout << " ";
          }
        }
        std::cout << "\n";
      }

      // Add separator line between block rows
      if (i < block_rows_ - 1 && with_separator) {
        // Print horizontal separator line
        for (std::size_t j = 0; j < block_cols_; ++j) {
          if (j > 0) std::cout << "+";
          for (std::size_t bj = 0; bj < bs; ++bj) {
            std::cout << std::string(field_width, '-');
            if (bj < bs - 1) std::cout << "-";
          }
        }
        std::cout << "\n";
      }
    }

    if (with_separator) std::cout << "--------------------------------------------------\n";

    // Restore original formatting
    std::cout.flags(original_flags);
    std::cout.precision(original_precision);
  }

private:
  std::shared_ptr<const gko::Executor> exec_;
  std::size_t block_rows_;
  std::size_t block_cols_;

  std::vector<std::vector<std::unique_ptr<gko::matrix::Dense<T>>>> block_views;

public:
  gko::array<T> data;
};
} // namespace trl::ginkgo
