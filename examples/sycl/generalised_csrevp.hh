#pragma once

#include "csrevp.hh"
#include "evp_base.hh"

#include <cuda_runtime.h>
#include <cudss.h>

template <class T, unsigned int bs>
class GeneralisedCSREVP : public StandardEVPBase<T, bs> {
public:
  using Base = StandardEVPBase<T, bs>;
  using BlockView = typename Base::BlockView;
  using MatrixBlockView = typename Base::BlockMultivector::BlockMatrix::BlockView;

  GeneralisedCSREVP(sycl::queue queue, const std::string& matrix_A_file, const std::string& matrix_B_file)
      : Base(queue, 0)
      , matrix_B(matrix_B_file, queue)
      , matrix_AsB(matrix_A_file, matrix_B_file, 1e-4, queue)
  {
    if (matrix_B.rows() != matrix_B.cols()) throw std::runtime_error("GeneralisedCSREVP requires square matrices");

    // Set the matrix dimension in the base class and reinitialize Vtemp
    this->N = matrix_B.rows();
    this->Vtemp.emplace(this->create_multivector(this->N, bs));

    cudssCreate(&handle);
    // cudaStream_t stream = sycl::get_native<sycl::backend::cuda>(queue);
    // cudssSetStream(handle, stream); // TODO: Doesn't compile with AdaptiveCpp
    cudssConfigCreate(&solverConfig);
    cudssDataCreate(handle, &solverData);

    auto mtype = CUDSS_MTYPE_GENERAL;
    auto mview = CUDSS_MVIEW_FULL;
    auto base = CUDSS_BASE_ZERO;
    cudssMatrixCreateCsr(&Asb, matrix_AsB.rows(), matrix_AsB.cols(), matrix_AsB.nnz(), (void*)matrix_AsB.rowOffsets(), nullptr,
                         (void*)matrix_AsB.colIndices(), (void*)matrix_AsB.values(), CUDA_R_32I, CUDA_R_64F, mtype, mview, base);

    auto Xv = this->create_multivector(this->N, bs);
    auto Bv = this->create_multivector(this->N, bs);

    cudssMatrix_t X;
    cudssMatrix_t B;
    cudssMatrixCreateDn(&X, this->N, bs, bs, Xv.block_view(0).data, CUDA_R_64F, CUDSS_LAYOUT_ROW_MAJOR);
    cudssMatrixCreateDn(&B, this->N, bs, bs, Bv.block_view(0).data, CUDA_R_64F, CUDSS_LAYOUT_ROW_MAJOR);

    cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, Asb, X, B);
    cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, Asb, X, B);

    // cudssMatrixDestroy(X);
    // cudssMatrixDestroy(B);
  }

  void dot(BlockView X, BlockView Y, MatrixBlockView Z) override
  {
    auto V0 = this->Vtemp->block_view(0);

    applyB(X, V0);
    V0.dot(Y, Z);
  }

  void apply(BlockView X, BlockView Y) override
  {
    auto V0 = this->Vtemp->block_view(0);
    applyB(X, V0);

    this->queue.submit([&, Asb = Asb, handle = handle, solverConfig = solverConfig, solverData = solverData](auto& cgh) {
      auto* Bdata = V0.data;
      auto* Xdata = Y.data;
      int N = this->N;

      cgh.AdaptiveCpp_enqueue_custom_operation([=](auto& ih) {
        auto stream = ih.template get_native_queue<sycl::backend::cuda>();
        cudssSetStream(handle, stream);

        cudssMatrix_t X;
        cudssMatrix_t B;
        cudssMatrixCreateDn(&X, N, bs, bs, Xdata, CUDA_R_64F, CUDSS_LAYOUT_ROW_MAJOR);
        cudssMatrixCreateDn(&B, N, bs, bs, Bdata, CUDA_R_64F, CUDSS_LAYOUT_ROW_MAJOR);

        cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, Asb, X, B);

        // cudssMatrixDestroy(X);
        // cudssMatrixDestroy(B);
      });
    });
  }

private:
  void applyB(typename Base::BlockView X, typename Base::BlockView Y)
  {
    static auto* ev = trl::sycl::SyclProfiler::get().registerOrGetEvent(trl::sycl::SyclProfiler::get().registerOrGetFamily("CSREVP"), "applyB");

    const auto ncus = this->queue.get_device().template get_info<::sycl::info::device::max_compute_units>();
    const auto global_size = 128 * bs * ncus;

    T* X_data = X.data;
    T* Y_data = Y.data;
    const int* row_offsets = matrix_B.rowOffsets();
    const int* col_indices = matrix_B.colIndices();
    const T* values = matrix_B.values();
    const std::size_t num_rows = matrix_B.rows();

    auto e = this->queue.parallel_for(sycl::range<1>(global_size), [=](sycl::id<1> idx) {
      const std::size_t start = idx[0];

      for (std::size_t row = start; row < num_rows; row += global_size) {
        const int row_start = row_offsets[row];
        const int row_end = row_offsets[row + 1];

        // Initialize output row to zero
        for (std::size_t j = 0; j < bs; ++j) Y_data[row * bs + j] = T(0);

        // Accumulate contributions from all nonzeros in this row
        for (int k = row_start; k < row_end; ++k) {
          const std::size_t col = col_indices[k];
          const T val = static_cast<T>(values[k]);

          for (std::size_t j = 0; j < bs; ++j) Y_data[row * bs + j] += val * X_data[col * bs + j];
        }
      }
    });
    trl::sycl::SyclProfiler::get().pushEvent(ev, e);
  }

  CSRMatrix<T> matrix_B;
  CSRMatrix<T> matrix_AsB;

  // cuDSS data
  cudssHandle_t handle;
  cudssConfig_t solverConfig;
  cudssData_t solverData;
  cudssMatrix_t Asb;
};
