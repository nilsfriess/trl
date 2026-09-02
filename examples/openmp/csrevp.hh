#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>

#include "trl/common.hh"
#include "trl/openmp/backend.hh"

// CSR-based eigenvalue problem
template <class T, unsigned int bs>
class StandardEVPOperator : public trl::EuclideanDot<trl::openmp::Backend<T, bs>> {
public:
  using Backend = trl::openmp::Backend<T, bs>;

  StandardEVPOperator(const std::string& matrix_file)
  {
    Eigen::loadMarket(A, matrix_file);

    if (A.rows() != A.cols()) throw std::runtime_error("CSREVP requires a square matrix");
    N = A.rows();
  }

  void apply(typename Backend::Multivector::BlockView X, typename Backend::Multivector::BlockView Y)
  {
    constexpr auto storage = bs == 1 ? Eigen::ColMajor : Eigen::RowMajor;

    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, bs, storage>> Xmap(X.data_, A.rows(), bs);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, bs, storage>> Ymap(Y.data_, A.rows(), bs);

    Ymap = A.template selfadjointView<Eigen::Lower>() * Xmap;
  }

  std::size_t size() const { return N; }

private:
  Eigen::SparseMatrix<T> A;
  std::size_t N{};
};

template <class T, unsigned int bs>
class GeneralizedEVPOperator {
  using SparseMatrix = Eigen::SparseMatrix<T>;
  using Solver = Eigen::SparseLU<SparseMatrix>;
  constexpr static auto storage = bs == 1 ? Eigen::ColMajor : Eigen::RowMajor;

public:
  using Backend = trl::openmp::Backend<T, bs>;

  GeneralizedEVPOperator(const std::string& matrix_file_A, const std::string& matrix_file_B, double shift)
  {
    SparseMatrix A;
    Eigen::loadMarket(A, matrix_file_A);
    Eigen::loadMarket(B, matrix_file_B);
    if ((A.rows() != A.cols()) or (B.rows() != B.cols())) throw std::runtime_error("Matrices must be square");

    using SpMat = typename SparseMatrix::PlainObject;
    SpMat matA = A.template selfadjointView<Eigen::Lower>();
    SpMat matB = B.template selfadjointView<Eigen::Lower>();
    SpMat mat = matA - shift * matB;
    solver.isSymmetric(true);
    solver.compute(mat);
  }

  void apply(typename Backend::Multivector::BlockView X, typename Backend::Multivector::BlockView Y)
  {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, bs, storage>> Xmap(X.data_, B.rows(), bs);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, bs, storage>> Ymap(Y.data_, B.rows(), bs);

    tmp = B.template selfadjointView<Eigen::Lower>() * Xmap;

    // SparseLU::solve requires a column-major destination, but the multivector
    // blocks are row-major for bs > 1, so solve into a temp and copy over.
    Ytmp = solver.solve(tmp);
    Ymap = Ytmp;
  }

  void dot(typename Backend::Multivector::BlockView V, typename Backend::Multivector::BlockView W, typename Backend::BlockMatrix::BlockView R)
  {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, bs, storage>> Wmap(W.data_, B.rows(), bs);

    tmp = B.template selfadjointView<Eigen::Lower>() * Wmap;

    typename Backend::Multivector::BlockView Tview(tmp.data(), B.rows());
    V.dot(Tview, R);
  }

  std::size_t size() const { return B.rows(); }

private:
  Solver solver;
  SparseMatrix B;

  Eigen::Matrix<T, Eigen::Dynamic, bs, storage> tmp;
  Eigen::Matrix<T, Eigen::Dynamic, bs, Eigen::ColMajor> Ytmp; // column-major staging buffer for the LU solve
};
