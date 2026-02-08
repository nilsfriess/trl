#include <iostream>

#include "../test_lanczos_extend.hh"
#include "diagonal.hh"
#include "test_helper.hh"

using trl::openmp::tests::DiagonalEVP;

template <class Scalar, unsigned int bs>
bool run_test(bool verbose)
{
  const unsigned int N = 128;
  using EVP = DiagonalEVP<Scalar, bs>;

  auto evp = std::make_shared<EVP>(N);
  Scalar tolerance = 1e-8;

  OpenMPTestHelper<EVP> helper;
  return test_lanczos_extend(evp, helper, tolerance, verbose);
}

int main()
{
  bool verbose = true;

  std::cout << "========================================\n";
  std::cout << "<<<<<<<<<      OpenMP TEST       >>>>>>>\n";
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
