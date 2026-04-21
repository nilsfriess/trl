// Example: Compute largest eigenvalues of a sparse matrix from a Matrix Market file
//
// Usage: ./csrtest <matrix.mtx> [nev] [ncv]
//   matrix.mtx: Path to a Matrix Market file containing a symmetric sparse matrix
//   nev: Number of eigenvalues to compute (default: 16)
//   ncv: Number of Lanczos vectors (default: 4 * nev)

#include "../csrtest_common.hh"
#include "csrevp.hh"

constexpr unsigned int BLOCKSIZE = 4;

int main(int argc, char* argv[])
{
  using EVP = CSREVP<double, BLOCKSIZE>;

  return run_csrtest<BLOCKSIZE>(
      argc,
      argv,
      [](const std::string& matrix_file) { return std::make_shared<EVP>(matrix_file); },
      []() {},
      [](auto V0, auto& rng, auto& dist) { std::generate_n(V0.data_, V0.rows() * V0.cols(), [&]() { return dist(rng); }); },
      []() {});
}
