#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>
#include <trl/impl/sycl/multivector.hh>

int main(int argc, char** argv)
{
  using Scalar = double;
  constexpr unsigned int bs = 8;

  const std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : (1 << 20);
  const int iters = (argc > 2) ? std::atoi(argv[2]) : 100;

  sycl::queue q{sycl::property::queue::in_order{}};

  trl::sycl::BlockMultivector<Scalar, bs> V(q, n, bs);
  trl::sycl::BlockMultivector<Scalar, bs> W(q, n, bs);
  trl::sycl::BlockMatrix<Scalar, bs> R(q, 1, 1);

  auto V0 = V.block_view(0);
  auto W0 = W.block_view(0);
  auto R00 = R.block_view(0, 0);

  std::mt19937 rng(42);
  std::uniform_real_distribution<Scalar> dist(Scalar(-1), Scalar(1));
  {
    std::vector<Scalar> host(n * bs);
    std::generate_n(host.begin(), n * bs, [&]() { return dist(rng); });
    q.memcpy(V0.data, host.data(), n * bs * sizeof(Scalar));
    std::generate_n(host.begin(), n * bs, [&]() { return dist(rng); });
    q.memcpy(W0.data, host.data(), n * bs * sizeof(Scalar));
    q.wait();
  }

  for (int i = 0; i < iters; ++i) V0.dot(W0, R00);
  q.wait();

  Scalar checksum = 0;
  for (std::size_t i = 0; i < bs * bs; ++i) checksum += R00.data[i];

  std::cout << "dot checksum: " << checksum << "\n";
  return 0;
}
