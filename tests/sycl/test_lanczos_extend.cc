#include <iostream>

#include "../test_lanczos_extend.hh"
#include "sycl/diagonal.hh"
#include "test_helper.hh"

#include <memory>
#include <sycl/sycl.hpp>

template <class Scalar, unsigned int bs>
bool run_test(sycl::queue q, bool verbose)
{
  const unsigned int N = 512;
  using EVP = DiagonalEVP<Scalar, bs>;

  auto evp = std::make_shared<EVP>(q, N);
  double tolerance = 1e-8;

  SYCLTestHelper<EVP> helper(q);
  return test_lanczos_extend(evp, helper, tolerance, verbose);
}

int main()
{
  bool verbose = true;

  std::cout << "========================================\n";
  std::cout << "<<<<<<<        SYCL TEST        >>>>>>>>\n";
  std::cout << "========================================\n";

  std::cout << "========================================\n";
  std::cout << "Testing with DiagonalEVP\n";
  std::cout << "========================================\n";

  int num_failed = 0;

  sycl::queue q{sycl::property::queue::in_order()};
  if (!run_test<double, 1>(q, verbose)) num_failed++;
  if (!run_test<double, 2>(q, verbose)) num_failed++;
  if (!run_test<double, 4>(q, verbose)) num_failed++;
  if (!run_test<double, 8>(q, verbose)) num_failed++;

  return num_failed;
}
