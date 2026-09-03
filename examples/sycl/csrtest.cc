// Example: Compute largest eigenvalues of a sparse matrix from a Matrix Market file
//
// Usage: ./csrtest_sycl <matrix.mtx> [nev] [ncv] [num_runs]

#include <iostream>
#include <memory>

#include <sycl/sycl.hpp>

#include "../csr_driver.hh"
#include "csrevp.hh"
#include "trl/sycl/backend.hh"

constexpr unsigned int BLOCKSIZE = 1;

int main(int argc, char* argv[])
{
  trl::examples::CsrTestOptions opts;
  if (!trl::examples::parse_args(argc, argv, opts)) return 1;

  try {
    sycl::queue queue{sycl::property::queue::in_order{}};
    std::cout << "SYCL device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n\n";

    trl::Sycl::Backend<double, BLOCKSIZE> backend(queue);
    auto op = std::make_shared<CSREVP<double, BLOCKSIZE>>(queue, opts.matrix_file);

    return trl::examples::run_csr_test(backend, op, opts);
  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
