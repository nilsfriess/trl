#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>

#include "trl/impl/openmp/evp_base.hh"
#include "trl/impl/openmp/util.hh"

// CSR-based eigenvalue problem that inherits from EVPBase
template <class T, unsigned int bs>
class CSREVP : public trl::openmp::EVPBase<T, bs> {
public:
  using Base = trl::openmp::EVPBase<T, bs>;

  CSREVP(const std::string& matrix_file)
      : Base(0) // N will be set after loading
  {
    Eigen::loadMarket(A, matrix_file);

    if (A.rows() != A.cols()) throw std::runtime_error("CSREVP requires a square matrix");

    // Set the matrix dimension in the base class and reinitialize Vtemp
    this->N = A.rows();
    this->Vtemp.emplace(this->create_multivector(this->N, bs));
  }

  void apply(typename Base::BlockView X, typename Base::BlockView Y)
  {
    constexpr auto storage = bs == 1 ? Eigen::ColMajor : Eigen::RowMajor;

    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, bs, storage>> Xmap(X.data_, A.rows(), bs);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, bs, storage>> Ymap(Y.data_, A.rows(), bs);

    Ymap = A.template selfadjointView<Eigen::Lower>() * Xmap;
  }

private:
  Eigen::SparseMatrix<T> A;
};

template <class T, unsigned int bs>
class CSRGeneralizedEVP : public trl::openmp::EVPBase<T, bs> {
  using SparseMatrix = Eigen::SparseMatrix<T>;
  using Solver = Eigen::SparseLU<SparseMatrix>;
  constexpr static auto storage = bs == 1 ? Eigen::ColMajor : Eigen::RowMajor;

public:
  using Base = trl::openmp::EVPBase<T, bs>;

  CSRGeneralizedEVP(const std::string& matrix_file_A, const std::string& matrix_file_B, double shift)
      : Base(0) // N will be set after loading
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

    // Set the matrix dimension in the base class and reinitialize Vtemp
    this->N = A.rows();
    this->Vtemp.emplace(this->create_multivector(this->N, bs));
    tmp.resize(this->N, bs);
  }

  void apply(typename Base::BlockView X, typename Base::BlockView Y) override
  {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, bs, storage>> Xmap(X.data_, B.rows(), bs);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, bs, storage>> Ymap(Y.data_, B.rows(), bs);

    tmp = B.template selfadjointView<Eigen::Lower>() * Xmap;
    Ymap = solver.solve(tmp);
  }

  void dot(typename Base::BlockView V, typename Base::BlockView W, typename Base::BlockMatrixBlockView R) override
  {
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, bs, storage>> Wmap(W.data_, B.rows(), bs);

    tmp = B.template selfadjointView<Eigen::Lower>() * Wmap;

    typename Base::BlockView Tview(tmp.data(), B.rows());
    V.dot(Tview, R);
  }

private:
  Solver solver;
  SparseMatrix B;

  Eigen::Matrix<T, Eigen::Dynamic, bs, storage> tmp;
};
