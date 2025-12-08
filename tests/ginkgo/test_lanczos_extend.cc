#include <ginkgo/ginkgo.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

#include <trl/concepts.hh>
#include <trl/eigensolvers/lanczos.hh>
#include <trl/eigensolvers/params.hh>

#include "../test_lanczos_extend.hh"
#include "ginkgo/diagonal.hh"
#include "ginkgo_test_helper.hh"

template <class Scalar, unsigned int bs>
bool run_test(bool verbose)
{
  auto exec = gko::OmpExecutor::create();

  const unsigned int N = 256;
  using EVP = DiagonalEigenproblem<Scalar, bs>;
  GinkgoTestHelper<EVP> helper;
  auto evp = std::make_shared<EVP>(exec, N);
  double tolerance = 1e-8;

  return test_lanczos_extend(evp, helper, tolerance, verbose);
}

int main()
{
  bool verbose = true;

  std::cout << "========================================\n";
  std::cout << "<<<<<<<       GINKGO TEST       >>>>>>>>\n";
  std::cout << "========================================\n";

  std::cout << "========================================\n";
  std::cout << "Testing with DiagonalEVP\n";
  std::cout << "========================================\n";

  int num_failed = 0;

  if (!run_test<double, 1>(verbose)) num_failed++;
  if (!run_test<double, 2>(verbose)) num_failed++;
  if (!run_test<double, 4>(verbose)) num_failed++;
  if (!run_test<double, 8>(verbose)) num_failed++;

  return num_failed;
}
