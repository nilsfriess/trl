// Example: Compute largest eigenvalues of a sparse matrix from a Matrix Market file
//
// Usage: ./csrtest_openmp <matrix.mtx> [nev] [ncv] [num_runs]

#include <iostream>
#include <memory>

#include "../csr_driver.hh"
#include "csrevp.hh"
#include "trl/openmp/backend.hh"

constexpr unsigned int BLOCKSIZE = 1;

int main(int argc, char* argv[])
{
  trl::examples::CsrTestOptions opts;
  if (!trl::examples::parse_args(argc, argv, opts)) return 1;

  try {
    trl::openmp::Backend<double, BLOCKSIZE> backend;
    auto op = std::make_shared<StandardEVPOperator<double, BLOCKSIZE>>(opts.matrix_file);

    return trl::examples::run_csr_test(backend, op, opts);
  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
