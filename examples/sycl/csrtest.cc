// Example: Compute largest eigenvalues of a sparse matrix from a Matrix Market file
//
// Usage: ./csrtest <matrix.mtx> [nev] [ncv]
//   matrix.mtx: Path to a Matrix Market file containing a symmetric sparse matrix
//   nev: Number of eigenvalues to compute (default: 10)
//   ncv: Number of Lanczos vectors (default: 4 * nev)

#include "../csrtest_common.hh"
#include "csrevp.hh"
#include "trl/impl/sycl/profiling.hh"

#include <sycl/sycl.hpp>
#include <vector>

constexpr unsigned int BLOCKSIZE = 1;

int main(int argc, char* argv[])
{
  using Scalar = float;

  sycl::queue queue{{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
  using EVP = CSREVP<Scalar, BLOCKSIZE>;

  const int result = run_csrtest<BLOCKSIZE>(
      argc, argv, [&](const std::string& matrix_file) { return std::make_shared<EVP>(queue, matrix_file); },
      [&]() { std::cout << "SYCL device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n\n"; },
      [&](auto V0, auto& rng, auto& dist) {
        const std::size_t n = V0.rows() * V0.cols();
        std::vector<Scalar> v0_host(n);
        std::generate_n(v0_host.begin(), n, [&]() { return static_cast<Scalar>(dist(rng)); });
        queue.memcpy(V0.data, v0_host.data(), n * sizeof(Scalar)).wait();
      },
      [&]() { queue.wait(); });

  trl::sycl::SyclProfiler::get().report();
  return result;
}
