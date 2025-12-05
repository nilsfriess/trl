#if WITH_SYCL_BACKEND
#include <sycl/sycl.hpp>

#include "sycl/diagonal.hh"
#endif

#if WITH_GINKGO_BACKEND
#include "ginkgo/laplace.hh"

#include <ginkgo/ginkgo.hpp>
#endif

#include <cassert>
#include <cmath>
#include <iostream>

#if WITH_SYCL_BACKEND
void test_diagonal_orthonormalize_sycl()
{
  sycl::queue q;
  const int N = 8;
  DiagonalEVP<double, 1> evp(q, N);
  auto V = evp.create_multivector(N, 1);
  auto V0 = V.block_view(0);
  for (int i = 0; i < N; ++i) V0.data[i] = 1.0;
  double* r_data = new double[1]{0.0};
  trl::MatrixBlockView<double, 1> R(&q, r_data);
  evp.orthonormalize(V0, R);
  // After orthonormalization, V0 should be normalized
  double norm = 0.0;
  for (int i = 0; i < N; ++i) norm += V0.data[i] * V0.data[i];
  norm = std::sqrt(norm);
  assert(std::abs(norm - 1.0) < 1e-12);
  delete[] r_data;
}
#endif

#if WITH_GINKGO_BACKEND
bool test_diagonal_orthonormalize_ginkgo()
{
  auto exec = gko::OmpExecutor::create();

  const auto tol = 1e-8;

  const unsigned int N = 256;
  const unsigned int bs = 4;
  const unsigned int cols = 8;
  DiagonalEigenproblem<double, bs> evp(exec, N);

  auto V = evp.create_multivector(N, bs);
  auto V0 = V.block_view(0);

  // const auto print_V0 = [&]() {
  //   for (std::size_t i = 0; i < V0.rows(); ++i) {
  //     for (std::size_t j = 0; j < V0.cols(); ++j) {
  //       auto entry = V0.data->at(i, j);
  //       std::cout << entry << " ";
  //     }
  //     std::cout << "\n";
  //   }
  //   std::cout << "\n";
  // };

  auto R = evp.create_blockmatrix(1, 1);
  auto R0 = R.block_view(0, 0);

  std::mt19937 gen(42);
  std::normal_distribution<double> dist(0.0, 1.0);
  for (std::size_t i = 0; i < V0.rows(); ++i)
    for (std::size_t j = 0; j < V0.cols(); ++j) V0.data->at(i, j) = dist(gen);

  evp.orthonormalize(V0, R0);

  auto W = evp.create_multivector(N, bs);
  auto W0 = W.block_view(0);
  W0.copy_from(V0);

  // After orthonormalization, V0 should be normalised w.r.t. the inner product defined by the EVP
  evp.dot(V0, W0, R0);

  bool passed = true;
  for (std::size_t i = 0; i < bs; ++i) {
    for (std::size_t j = 0; j < bs; ++j) {
      auto entry = R0.data->at(i, j);
      if (i == j && std::abs(entry - 1) > tol) passed = false;
      if (i != j && std::abs(entry) > tol) passed = false;
    }
  }
  return passed;
}
#endif

int main()
{
#if WITH_SYCL_BACKEND
  test_diagonal_orthonormalize_sycl();
  std::cout << "Info: SYCL test_diagonal_orthonormalize passed\n";
#else
  std::cout << "Warning: SYCL backend not available, test skipped.\n";
#endif

#if WITH_GINKGO_BACKEND
  bool success = test_diagonal_orthonormalize_ginkgo();
  if (success) std::cout << "Info: Ginkgo test_diagonal_orthonormalize passed\n";
  else std::cout << "Error: Ginkgo test_diagonal_orthonormalize failed\n";
#else
  std::cout << "Warning: Ginkgo backend not available, test skipped.\n";
#endif

  return 0;
}
