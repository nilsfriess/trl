// Example: Compute largest eigenvalues of a sparse matrix from a Matrix Market file
//
// Usage: ./csrtest <matrix.mtx> [nev] [ncv]
//   matrix.mtx: Path to a Matrix Market file containing a symmetric sparse matrix
//   nev: Number of eigenvalues to compute (default: 10)
//   ncv: Number of Lanczos vectors (default: 4 * nev)

#include <sycl/sycl.hpp>

#include "../csrtest_common.hh"
#include "csrevp.hh"

constexpr unsigned int BLOCKSIZE = 4;

int main(int argc, char* argv[])
{
  sycl::queue queue{sycl::property::queue::in_order{}};
  using EVP = CSREVP<double, BLOCKSIZE>;

  return run_csrtest<BLOCKSIZE>(
      argc,
      argv,
      [&](const std::string& matrix_file) { return std::make_shared<EVP>(queue, matrix_file); },
      [&]() { std::cout << "SYCL device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n\n"; },
      [](auto V0, auto& rng, auto& dist) { std::generate_n(V0.data, V0.rows() * V0.cols(), [&]() { return dist(rng); }); },
      [&]() { queue.wait(); });
}
