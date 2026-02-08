#include <cstdlib>
#include <iostream>
#include <random>

#include <trl/impl/openmp/blockmatrix.hh>
#include <trl/impl/openmp/multivector.hh>

int main(int argc, char** argv)
{
  using Scalar = double;
  constexpr unsigned int bs = 8;

  const std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : (1 << 20);
  const int iters = (argc > 2) ? std::atoi(argv[2]) : 100;

  trl::openmp::BlockMultivector<Scalar, bs> V(n, bs);
  trl::openmp::BlockMultivector<Scalar, bs> W(n, bs);
  trl::openmp::BlockMatrix<Scalar, bs> R(1, 1);

  auto V0 = V.block_view(0);
  auto W0 = W.block_view(0);
  auto R00 = R.block_view(0, 0);

  std::mt19937 rng(42);
  std::uniform_real_distribution<Scalar> dist(Scalar(-1), Scalar(1));
  for (std::size_t i = 0; i < n * bs; ++i) {
    V0.data[i] = dist(rng);
    W0.data[i] = dist(rng);
  }

  for (int i = 0; i < iters; ++i) V0.dot(W0, R00);

  Scalar checksum = 0;
  for (std::size_t i = 0; i < bs * bs; ++i) checksum += R00.data[i];

  std::cout << "dot checksum: " << checksum << "\n";
  return 0;
}
