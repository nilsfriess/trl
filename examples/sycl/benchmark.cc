#include "trl/impl/sycl/multivector.hh"

#include <chrono>
#include <iostream>
#include <random>
#include <ratio>
#include <sycl/sycl.hpp>

struct RunParams {
  int n_warmup;
  int n_benchmark;
};

template <class T, unsigned blocksize, int worker_mult = 64>
std::pair<T, std::chrono::duration<double>> run(sycl::queue& queue, std::size_t n, const RunParams& run_params)
{
  using Multivec = trl::sycl::BlockMultivector<T, blocksize>;
  using BlockMatrix = typename Multivec::BlockMatrix;

  Multivec U(queue, n, blocksize);
  Multivec V(queue, n, blocksize);
  BlockMatrix R(queue, 1, 1);

  auto u = U.block_view(0);
  auto v = V.block_view(0);
  auto r = R.block_view(0, 0);

  std::minstd_rand0 rng(42);
  std::uniform_real_distribution<double> dist(-1, 1);
  std::generate_n(v.data, v.rows() * v.cols(), [&]() { return dist(rng); });
  std::generate_n(u.data, u.rows() * u.cols(), [&]() { return dist(rng); });

  for (int i = 0; i < run_params.n_warmup; ++i) u.template dot<worker_mult>(v, r);
  queue.wait();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < run_params.n_benchmark; ++i) u.template dot<worker_mult>(v, r);
  queue.wait();
  auto end = std::chrono::steady_clock::now();
  return {r.data[0], (end - start) / run_params.n_benchmark};
}

auto flops(std::size_t n, unsigned int m) { return 2 * n * m * m; }

template <class Scalar, unsigned bs, int worker_mult>
void measure(sycl::queue& queue, std::size_t n, const RunParams& params)
{
  auto [data, time] = run<Scalar, bs, worker_mult>(queue, n, params);
  std::cout << "  worker_mult=" << worker_mult << ": " << time.count() * 1000. << "ms, "
            << flops(n, bs) / time.count() / 1e9 << " GFLOP/s\n";
}

template <class Scalar, unsigned bs>
void measure_all_worker_mult(sycl::queue& queue, std::size_t n, const RunParams& params)
{
  std::cout << "Blocksize " << bs << ":\n";
  measure<Scalar, bs, 16>(queue, n, params);
  measure<Scalar, bs, 32>(queue, n, params);
  measure<Scalar, bs, 64>(queue, n, params);
  measure<Scalar, bs, 128>(queue, n, params);
  measure<Scalar, bs, 256>(queue, n, params);
  measure<Scalar, bs, 512>(queue, n, params);
}

template <class Scalar>
void test(sycl::queue& queue, std::size_t n, const RunParams& params)
{
  if constexpr (std::is_same_v<Scalar, double>) std::cout << "===== Running double tests =====\n";
  else std::cout << "===== Running float tests =====\n";

  measure_all_worker_mult<Scalar, 1>(queue, n, params);
  measure_all_worker_mult<Scalar, 2>(queue, n, params);
  measure_all_worker_mult<Scalar, 4>(queue, n, params);
  measure_all_worker_mult<Scalar, 8>(queue, n, params);
  measure_all_worker_mult<Scalar, 16>(queue, n, params);
}

int main()
{
  sycl::queue queue{sycl::property::queue::in_order{}};

  std::size_t n = 1 << 20;
  RunParams params{.n_warmup = 10, .n_benchmark = 100};

  test<double>(queue, n, params);
  test<float>(queue, n, params);
}
